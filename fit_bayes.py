#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:52:41 2024

@author: epowell
"""
# Software package for Bayesian hierarchical model fitting
# Copyright (C) 2023 Elizabeth Powell, Paddy Slator

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#=============================================================================#
#
# Run Bayesian fitting
#
# [params_final, acceptance_rate, param_conv] ...
#    = fit_bayes(model, acq_scheme, data, E_fit, params_init_vector,...
#                params_fixed, mask, nsteps, burn_in, nupdate, update_rois)
#
# INPUT
# - model:                  model structure
# - acq_scheme              acquisition scheme structure
# - data                    MRI data to perform fitting on
# - params_init_vector:     initial LSQ fit (original parameter scaling)
# - params_fixed:           cell containing names of fixed parameters (i.e. not updated during MCMC)
#                           - can be empty
# - mask:                   rois
# - nsteps                  no MCMC steps
# - burn_in                 no steps in burn-in
# - nupdate                 how often to update weights (e.g. nupdate = 100
#                           -> weights updated every 100 steps)
# - update_rois             voxel ROI membership updated every step [bool]
#                           NOTE: untested functionality
#
# OUTPUT
# - params_final:           HBM-derived model parameters
# - acceptance_rate:        acceptance rate over MCMC
# - param_conv:             posterior distribution
#
# Notes:
#
# Authors: Elizabeth Powell, 04/09/2023
#
#=============================================================================#

import numpy as np
from copy import copy, deepcopy
import scipy

# %%
def tform_params(param_dict, parameter_names, model, idx_roi=None, direction='f'):
    param_dict = deepcopy(param_dict)
    
    if idx_roi is None:
        idx_roi = np.arange(len(param_dict[parameter_names[0]]))
        to_ignore = np.zeros(len(param_dict[parameter_names[0]]), dtype=int)
        for param in parameter_names:
            if param != '_mu':
                to_ignore[np.isnan(param_dict[param])] = 1
        idx_roi = idx_roi[~to_ignore.astype(bool)]
    
    if direction == 'f':
        for param in parameter_names:
            if param != '_mu':
                lb = (model.parameter_ranges[param][0] - 1e-5) * model.parameter_scales[param]
                ub = (model.parameter_ranges[param][1] + 1e-5) * model.parameter_scales[param]
                param_dict[param][idx_roi] = np.log(param_dict[param][idx_roi] - lb) - np.log(ub - param_dict[param][idx_roi])

    elif direction == 'r':
        for param in parameter_names:
            if param != '_mu':
                lb = (model.parameter_ranges[param][0] - 1e-5) * model.parameter_scales[param]
                ub = (model.parameter_ranges[param][1] + 1e-5) * model.parameter_scales[param]
                if model.parameter_ranges[param][0] != model.parameter_ranges[param][1]:
                    param_dict[param][idx_roi] = (lb + ub * np.exp(param_dict[param][idx_roi])) / (1 + np.exp(param_dict[param][idx_roi]))
                    param_dict[param][idx_roi] = np.where(np.isnan(param_dict[param][idx_roi]), ub, param_dict[param][idx_roi])
                    param_dict[param][idx_roi] = np.where(np.isinf(param_dict[param][idx_roi]), lb, param_dict[param][idx_roi])
                else:
                    idx = np.where(~np.isnan(param_dict[param][idx_roi]))
                    param_dict[param][idx_roi[idx]] = lb
            if np.any(param_dict[param] == 0):
                pass

    else:
        print('Incorrect input! Nothing is happening...')
    
    for param in parameter_names:
        param_dict[param] = np.real(param_dict[param])

    return param_dict


# %%
def compute_log_likelihood(model, acq_scheme, signal, predicted_signal=None, parameter_struct=None):
    if (predicted_signal is None and parameter_struct is None) or (predicted_signal is not None and parameter_struct is not None):
        raise ValueError('Input EITHER "predicted_signal" OR "parameter_struct_new". Exiting...')

    ndw = len(acq_scheme.bvalues)

    # actual measured signal
    y = deepcopy(signal)

    # model-predicted signal (current params)
    if predicted_signal is not None and parameter_struct is None:
        g = predicted_signal  # use input 
    elif predicted_signal is None and parameter_struct is not None:
        g = np.array(model.simulate_signal(acq_scheme, parameter_struct))  # calculate

    # calculate posteriors and PDFs (log scale)
    inner_y_y = np.sum(y * y, axis=1)
    inner_y_g = np.sum(y * g, axis=1)
    inner_g_g = np.sum(g * g, axis=1)
    log_likelihood = (-ndw / 2) * np.log(inner_y_y - (inner_y_g**2 / inner_g_g))

    return log_likelihood, g


# %%
def logmvnpdf(x, mean, cov):
    """
    Compute the log of the multivariate normal PDF.

    Parameters:
        x (ndarray): Input data, shape (n, d) where `n` is the number of samples, `d` is the dimensionality.
        mean (ndarray): Mean vector of the multivariate normal distribution, shape (d,).
        cov (ndarray): Covariance matrix, shape (d, d).

    Returns:
        log_pdf (ndarray): Log of the multivariate normal PDF for each sample, shape (n,).
    """
    rv = scipy.stats.multivariate_normal(mean=mean, cov=cov)
    log_pdf = rv.logpdf(x)
    return log_pdf


# %%
def fit(model, acq_scheme, data, params_init_vector, params_fixed, 
        mask, nsteps, burn_in, nupdate, update_rois):
    
    # FIXME: need quick hack to allow for different relaxation times in different ROIs 
    
    # FIXME: data checks?
    print(' >> at start of fit_bayes.m')
    
    # ensure voxel data is in 1D array
    if data.ndim > 2:  # need array size [nvox x ndw]
        data = data.reshape(-1, acq_scheme.shape[0])
    
    if mask is not None and np.ndim(mask) > 1:  # need array size [nvox x 1]
        mask = np.ravel(mask)
    
    key = list(params_init_vector.keys())  # parameter names
    for k in range(len(key)):  # need array size [nvox x 1]
        if params_init_vector[key[k]].ndim > 1:
            params_init_vector[key[k]] = np.ravel(params_init_vector[key[k]])
    
    # set mask default
    if mask is None:
        mask = data[:, 0] > 0
    
    # scale parameters
    for p in range(len(model.parameter_names)):
        param = model.parameter_names[p]
        params_init_vector[param] *= model.parameter_scales[param]
        try:
            params_fixed[param] *= model.parameter_scales[param]
        except:
            pass
    
    # extract useful values
    nvox = len(mask)                                                            # total number of voxels
    nparams = np.sum([model.parameter_cardinality[param] for param in model.parameter_names])  # number of parameters
    ndw = len(acq_scheme.bvalues)
    ncomp = len(model.partial_volume_names)
    
    # extract ROIs if present in mask; check enough voxels in each ROI to avoid df error in sigma calculation
    print(' >> extracting ROIs')
    roi_vals = np.unique(mask[mask > 0])                                        # list of unique integers that identify each ROI (ignore 0's)
    roi_nvox = np.array([np.sum(mask == roi_val) for roi_val in roi_vals])      # no. voxels in each ROI
    toremove = roi_nvox < 2 * nparams                                           # remove ROIs with too few voxels
    roi_vals = roi_vals[~toremove]
    nroi = len(roi_vals)                                                        # no. remaining ROIs
    
    # get number of compartments; only fit no. compartments - 1
    print(' >> setting up dictionary')
    model_reduced = deepcopy(model)
    if model.partial_volume_names:
        # FIXME: untested
        parameters_to_fit = [param for param in model.parameter_names if param != model.partial_volume_names[-1]]
        dependent_fraction = model.partial_volume_names[-1]
        # remove dependent volume fraction from model
        model_reduced.parameter_names = [param for param in model_reduced.parameter_names if param != dependent_fraction]
        model_reduced.parameter_ranges.pop(dependent_fraction, None)
        model_reduced.parameter_cardinality.pop(dependent_fraction, None)
        model_reduced.parameter_scales.pop(dependent_fraction, None)
    else:
        parameters_to_fit = model.parameter_names
        dependent_fraction = ''
    
    # remove any parameters that have the same upper and lower bounds OR have the fix me flag
    for param in model_reduced.parameter_names:
        if model_reduced.parameter_ranges[param][0] == model_reduced.parameter_ranges[param][1] or model_reduced.parameter_fixed[param] == 1:
            # FIXME: untested
            model_reduced.parameter_names.remove(param)
            model_reduced.parameter_ranges.pop(param, None)
            model_reduced.parameter_cardinality.pop(param, None)
            model_reduced.parameter_scales.pop(param, None)
    
    parameters_to_fit = model_reduced.parameter_names
    nparams_red = np.sum([np.prod(model_reduced.parameter_cardinality[param]) for param in model_reduced.parameter_names])
    params_init = dict.fromkeys(model.parameter_names)
    for param in model.parameter_names:
        # FIXME: untested for model.parameter_cardinality[param] > 1
        params_init[param] = np.squeeze(np.full((nvox, model.parameter_cardinality[param]), np.nan))
        params_init[param][mask > 0, ] = params_init_vector[param][mask > 0]

    params_current = deepcopy(params_init)
        
    # log transform variables (non-orientation only) (original -> log)
    params_current_tform = tform_params(params_init, model.parameter_names, model, None, 'f')  # dict of initial LSQ fit values
    parameter_ranges_tform = {param: np.array(model.parameter_ranges[param]) * model.parameter_scales[param] for param in model.parameter_names}
    parameter_ranges_tform = tform_params(parameter_ranges_tform, model.parameter_names, model, None, 'f')
    
    # initial weights for Metropolis-Hastings parameter sampling
    print(' >> setting up weights')
    w = {param: np.array(model.parameter_ranges[param]) * model.parameter_scales[param] for param in parameters_to_fit}
    w = tform_params(w, parameters_to_fit, model, None, 'f')
    w_max = deepcopy(w)
    
    for param in parameters_to_fit:
        if model.parameter_cardinality[param] > 1:
            # FIXME: untested
            raise ValueError('Code conversion still needed re. parameter cardinality > 1 (1). Exiting...')
        elif model.parameter_cardinality[param] == 1 and param != 'partial_volume_0':
            w[param] = 0.001 * np.abs(w[param][1] - w[param][0])
            w_max[param] = 0.25 * np.abs(w_max[param][1] - w_max[param][0])
            w[param] = np.tile(w[param], (nvox, ))  # replicate to create weight for each voxel
            w_max[param] = np.tile(w_max[param], (nvox, ))
        elif model.parameter_cardinality[param] == 1 and param == 'partial_volume_0':
            # FIXME: untested
            w[param] = 0.001 * np.abs(w[param][1] - w[param][0])
            w_max[param] = 0.25 * np.abs(w_max[param][1] - w_max[param][0])
            w[param] = np.tile(w[param], (nvox, ))  # replicate to create weight for each voxel
            w_max[param] = np.tile(w_max[param], (nvox, ))
    
    # initialise variables to track state of optimisation at each step
    maxsize = 1e8  # limit max size of arrays (which is otherwise max nsteps*nvox)
    maxsteps = int(np.floor(maxsize / nvox))  # max no. updates to store
    whentostore = int(np.ceil(nsteps / maxsteps))  # ceil - undershoot max array size
    maxsteps = int(np.floor(nsteps / whentostore)) + 1  # need to recalculate because of rounding in previous line
    
    accepted = np.zeros(nvox, dtype=np.float32)  # total accepted moves over all steps
    accepted_per_n = np.zeros(nvox, dtype=np.float32)  # accepted moves per N steps (updates weights)
    acceptance_rate = np.zeros((nvox, maxsteps), dtype=np.float32)  # accepted moves at each step
    
    param_conv = {param: np.squeeze(np.zeros((nvox, model.parameter_cardinality[param], maxsteps), dtype=np.float32)) for param in parameters_to_fit}
    gibbs_mu = np.zeros((nroi, nparams_red, maxsteps))
    gibbs_sigma = np.zeros((nroi, nparams_red, nparams_red, maxsteps))
    gibbs_mu_current = np.zeros((nroi, nparams_red))
    gibbs_sigma_current = np.zeros((nroi, nparams_red, nparams_red))
    likelihood_stored = np.zeros((nvox, maxsteps), dtype=np.float32)
    
    w_stored = {param: np.squeeze(np.zeros((nvox, model.parameter_cardinality[param], int(np.floor(burn_in / (2 * nupdate))) + 1), dtype=np.float32)) for param in parameters_to_fit}
    
    mask_new = np.zeros((nvox, maxsteps), dtype=np.int8)  # ROI membership at each step
    params_final = dict.fromkeys(model.parameter_names)
    
    for param in parameters_to_fit:
        if model.parameter_cardinality[param] > 1:
            #FIXME: untested
            raise ValueError('Code conversion still needed re. parameter cardinality > 1 (5). Exiting...')
        elif model.parameter_cardinality[param] == 1:
            params_final[param] = np.zeros(nvox)
    
    # TESTING
    gibbs_mu_norm = np.zeros((nroi, nparams_red, maxsteps))  # gibbs mu at each step (normal space)
    gibbs_sigma_norm = np.zeros((nroi, nparams_red, nparams_red, maxsteps))  # gibbs sigma at each step (normal space)
    
    # initialise dictionaries (param_conv, accepted, accepted_per_n, acceptance_rate)
    print(f' >> no. weight updates = {int(np.floor(burn_in / nupdate)) + 1}')
    print(' >> initialising dictionaries')
    for param in parameters_to_fit:
        print(f'    > {param}')
        if model.parameter_cardinality[param] > 1:
            # FIXME: untested
            raise ValueError('Code conversion still needed re. parameter cardinality > 1 (2). Exiting...')
        else:
            if nupdate > 0:
                w_stored[param][:, 0] = w[param]

    #-----------------------------------------------------------------------#
    #------------------------------- MCMC ----------------------------------#
    print(" >> starting MCMC loop")
    
    # NB i (voxel loop) and j (MC loop) in keeping with Orton paper
    for j in range(0, nsteps):
        if j % 500 == 0: #j % 500 == 0:
            print(f"MCMC step = {j} / {nsteps}")
    
        # loop over ROIs
        for roi in range(nroi):
            idx_roi = np.where(mask == roi_vals[roi])[0]  # indices into mask of voxels in ROI
            nvox_roi = len(idx_roi)  # no. voxels in ROI
    
            # initialise sigma for this ROI
            if j == 0:
                sigma = np.cov(model_reduced.parameters_to_parameter_vector(**params_current_tform)[idx_roi], rowvar=False)
    
            #-------------------------------------------------------------#
            #                           TESTING                           #
            # do calculations in normal parameter ranges                  #
            params_current = tform_params(params_current_tform, model.parameter_names, model, None, 'r')
            params_current_vector = model_reduced.parameters_to_parameter_vector(**params_current)[idx_roi]
            if j == 0:
                sigma_norm = np.cov(model_reduced.parameters_to_parameter_vector(**params_current)[idx_roi], rowvar=False)
            #-------------------------------------------------------------#
    
            # FIXME: need quick hack to allow for different relaxation times in different ROIs
            # if len(acq_scheme_rois) > 1:
            #     acq_scheme = acq_scheme_rois[roi]
    
            # Gibbs moves to update priors.
            # Gibbs 1. sample mu from multivariate normal dist defined by current param estimates.
            # Add prior to hyperprior if defined
            params_current_tform_vector = model_reduced.parameters_to_parameter_vector(**params_current_tform)[idx_roi]
            m = np.mean(params_current_tform_vector, axis=0)
            V = sigma / nvox_roi
            mu = np.random.multivariate_normal(m, V)
    
            # Gibbs 2. sample sigma from inverse Wishart distribution (using newly updated mu)
            tmp = [np.outer(params_current_tform_vector[i, :] - mu,params_current_tform_vector[i, :] - mu) for i in range(params_current_tform_vector.shape[0])]
            phi = np.zeros((len(parameters_to_fit), len(parameters_to_fit)))
            for i in range(len(tmp)):
                phi += tmp[i]
            sigma = scipy.stats.invwishart(df=nvox_roi - nparams_red - 1, scale=phi).rvs()
    
            # save Gibbs parameters for this step (careful of parameter ordering)
            if j == 0:
                gibbs_mu[roi, :, j] = mu
                gibbs_sigma[roi, :, :, j] = sigma
            elif j % whentostore == 0:
                gibbs_mu[roi, :, j // whentostore] = mu
                gibbs_sigma[roi, :, :, j // whentostore] = sigma
    
            # save tmp version, needed for ROI membership update step
            gibbs_mu_current[roi, :] = mu
            gibbs_sigma_current[roi, :, :] = sigma
    
            #-------------------------------------------------------------#
            #                           TESTING                           #
            # do calculations in normal parameter ranges                  #
            m_norm = np.mean(params_current_vector, axis=0)
            V_norm = sigma_norm / nvox_roi
            mu_norm = np.random.multivariate_normal(m_norm, V_norm)
    
            # Gibbs 2. sample sigma from inverse Wishart distribution (using newly updated mu)
            tmp = [np.outer(params_current_vector[i, :] - mu_norm, params_current_vector[i, :] - mu_norm) for i in range(params_current_vector.shape[0])]
            phi_norm = np.zeros((len(parameters_to_fit), len(parameters_to_fit)))
            for i in range(len(tmp)):
                phi_norm += tmp[i]
    
            try:
                sigma_norm = scipy.stats.invwishart(df=nvox_roi - nparams_red - 1, scale=phi_norm).rvs()
            except Exception:
                sigma_norm = np.full((len(parameters_to_fit), len(parameters_to_fit)), np.nan)
    
            # save Gibbs parameters for this step (careful of parameter ordering)
            if j == 0:
                gibbs_mu_norm[roi, :, j] = mu_norm
                gibbs_sigma_norm[roi, :, :, j] = sigma_norm
            elif j % whentostore == 0:
                gibbs_mu_norm[roi, :, j // whentostore] = mu_norm
                gibbs_sigma_norm[roi, :, :, j // whentostore] = sigma_norm
            #-------------------------------------------------------------#
    
            # Metropolis-Hastings parameter updates
            params_new_tform = deepcopy(params_current_tform)
            for param in parameters_to_fit:
                # sample parameter
                if model.parameter_cardinality[param] > 1:
                    #FIXME: untested
                    for card in range(model["parameter_cardinality"][param]):
                        params_new_tform[param][idx_roi, card] = np.random.normal(
                            params_current_tform[param][idx_roi, card], w[param][idx_roi, card]
                        )
                elif model.parameter_cardinality[param] == 1:
                    params_new_tform[param][idx_roi] = np.random.normal(loc=params_current_tform[param][idx_roi], scale=np.matrix.flatten(w[param][idx_roi]))
                # FIXME (23/08/2024) if a volume fraction was sampled, 
                # re-compute dependent fraction (only super important if >3 volume fractions)
                # create boolean prior to avoid sum(volume fractions) > 1
    
            # compute acceptance
            params_new = tform_params(params_new_tform, model.parameter_names, model, None, 'r')  # transform parameters to normal
            likelihood = compute_log_likelihood(model, acq_scheme, data[idx_roi, :], None, model.parameters_to_parameter_vector(**params_current)[idx_roi, :])[0]
            likelihood_new = compute_log_likelihood(model, acq_scheme, data[idx_roi, :], None, model.parameters_to_parameter_vector(**params_new)[idx_roi])[0]
    
            params_current_tform_vector = model_reduced.parameters_to_parameter_vector(**params_current_tform)[idx_roi]
            prior = logmvnpdf(x=params_current_tform_vector, mean=mu, cov=sigma)
            params_new_tform_vector = model_reduced.parameters_to_parameter_vector(**params_new_tform)[idx_roi]
            prior_new = logmvnpdf(x=params_new_tform_vector, mean=mu, cov=sigma)
    
            alpha = np.array([(likelihood_new[i] + prior_new[i]) - (likelihood[i] + prior[i]) for i in range(nvox_roi)])
            r = np.log(np.random.rand(nvox_roi))
    
            # accept new parameter value if criteria met
            to_accept = np.column_stack((np.where(r < alpha)[0], idx_roi[r < alpha]))
            to_reject = np.column_stack((np.where(r >= alpha)[0], idx_roi[r >= alpha]))
    
            if to_accept.size > 0:  # account for error thrown by no accepted moves
                accepted[to_accept[:, 1]] += 1
                accepted_per_n[to_accept[:, 1]] += 1
                if j == 0:
                    likelihood_stored[to_accept[:, 1], j] = likelihood_new[to_accept[:, 0]] + prior_new[to_accept[:, 0]]
                elif j % whentostore == 0:
                    likelihood_stored[to_accept[:, 1], j // whentostore] = likelihood_new[to_accept[:, 0]] + prior_new[to_accept[:, 0]]
    
            if to_reject.size > 0:  # account for error thrown by all moves accepted
                if j == 0:
                    likelihood_stored[to_reject[:, 1], j] = likelihood[to_reject[:, 0]]
                elif j % whentostore == 0:
                    likelihood_stored[to_reject[:, 1], j // whentostore] = likelihood[to_reject[:, 0]]
    
            if j == 0:
                acceptance_rate[idx_roi, j] = accepted[idx_roi]
            elif j % whentostore == 0:
                acceptance_rate[idx_roi, j // whentostore] = accepted[idx_roi] / j
    
            # update current parameters (based on accepted / rejected)
            for param in parameters_to_fit:
                if model.parameter_cardinality[param] > 1:
                    #FIXME: untested
                    raise NotImplementedError("Code conversion still needed re. parameter cardinality > 1 (3). Exiting...")
                elif model.parameter_cardinality[param] == 1:
                    if to_accept.size > 0:  # account for error thrown by no accepted moves
                        params_current_tform[param][to_accept[:, 1]] = params_new_tform[param][to_accept[:, 1]]
    
                    if j == 0:
                        param_conv[param][idx_roi, j] = tform_params(params_current_tform, model_reduced.parameter_names, model, None, 'r')[param][idx_roi]
                    elif j % whentostore == 0:
                        param_conv[param][idx_roi, j // whentostore] = tform_params(params_current_tform, model_reduced.parameter_names, model, None, 'r')[param][idx_roi]
    
    
    
    
    
    
    
            # update weights
            if j % nupdate == 0 and 0 < j <= burn_in // 2:
                # hack to avoid weight=inf/0 if none/all moves in the last nupdate steps kept
                accepted_per_n[idx_roi] = np.minimum(accepted_per_n[idx_roi], nupdate * np.ones_like(accepted_per_n[idx_roi]))
                for param in parameters_to_fit:
                    if model.parameter_cardinality[param] > 1:
                        #FIXME: untested
                        raise NotImplementedError("Code conversion still needed re. parameter cardinality > 1 (4). Exiting...")
                    elif model.parameter_cardinality[param] == 1:
                        w[param][idx_roi] = np.minimum(w_max[param][idx_roi], (w[param][idx_roi] * (nupdate + 1)) / ((1 / (1 - 0.25)) * ((nupdate + 1) - accepted_per_n[idx_roi])))
                        w_stored[param][idx_roi, j // nupdate] = w[param][idx_roi]
                accepted_per_n[idx_roi] = np.zeros(nvox_roi)
        
            # end of roi loop
        
        
        
        # sum parameter estimates (divide by burn_in after mcmc loop) (saves memory - avoids saving every step individually) 
        if j > burn_in:
            params_current = tform_params(params_current_tform, model.parameter_names, model, None, 'r')
            for param in parameters_to_fit:
                if model.parameter_cardinality[param] > 1:
                    #FIXME: untested
                    raise ValueError('Code conversion still needed re. parameter cardinality > 1 (5). Exiting...')
                elif model.parameter_cardinality[param] == 1:
                    params_final[param] += params_current[param]

        # end of MCMC step loop

    # get mean of parameter estimates (divide sum by burn_in)
    for param in parameters_to_fit:
        if model.parameter_cardinality[param] > 1:
            #FIXME: untested
            raise ValueError('Code conversion still needed re. parameter cardinality > 1 (5). Exiting...')
        elif model.parameter_cardinality[param] == 1:
            params_final[param] = params_final[param] / (nsteps - burn_in)
    
    # add back the fixed parameters / dependent volume fractions
    toaddback = [param for param in model.parameter_names if param not in parameters_to_fit]
    #for p in range(len(toaddback)):
    #    param = toaddback[p]
    for param in toaddback:
        params_final[param] = params_init[param]

    return params_final, acceptance_rate, param_conv
