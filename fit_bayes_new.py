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

#==========================================================================
#
# Run Bayesian fitting
#
# [params_all, acceptance_rate, param_conv, likelihood_stored,...
#           w_stored, gibbs_mu, gibbs_sigma, gibbs_mu_norm, gibbs_sigma_norm] ...
#           = fit_bayes_new(model, acq_scheme, data, E_fit, params_init_vector,...
#                           params_fixed, sigma_constraint, mask, nsteps,...
#                           burn_in, nupdate, update_rois)
#
# INPUT
# - model:                  model structure
# - acq_scheme
# - data
# - E_fit
# - params_init_vector:     initial LSQ fit (original parameter scaling)
# - params_fixed:           cell containing names of fixed parameters (i.e. not updated during MCMC)
#                           - can be empty
# - sigma_con:              [nparams x 1] constrain sigma by proportion of parameter range 
#                           - can be empty
# - sigma0:                 prior for sigma hyperprior
# - mu0:                    prior for mu hyperprior
# - mask:                   rois
# - nsteps 
# - burn_in
# - nupdate

# - signal:                 signal (actual measured signal)
# - predicted_signal:       [opt] signal (predicted from current parameters)
# - parameter_struct_new:   [opt] current parameters
#
# OUTPUT
# - log_likelihood:         likelihood given current parameters
# - g:                      signal (predicted from current parameters)
#
# Notes:
# - input EITHER "predicted_signal" OR "parameter_struct_new"
#
# Authors: E Powell, 04/09/2023
#
#==========================================================================

import numpy as np
from copy import copy, deepcopy
import scipy
import matplotlib.pyplot as plt
from contextlib import contextmanager
import sys, os


# To temporarily suppress output to console
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# NOTE: will need fix if other models have parameters with cardinality > 1 (other than orientation)
def tform_params(param_dict, parameter_names, model, idx_roi, direction):
    param_dict = deepcopy(param_dict)  # because dicts are mutable, and don't want to alter inside function
    
    if idx_roi is None:
        idx_roi = np.array(range(0,param_dict[parameter_names[0]].__len__()))
        # remove background NaNs (if exist)
#        to_remove = [];
#        for param in parameter_names:
#            if '_mu' not in param:  # don't transform orientation parameters
#                [param_dict[param][x] for x in range(param_dict[param].__len__())]
#                to_remove = np.append(to_remove, [x for x in range(param_dict[param].__len__()) if np.isnan(param_dict[param][x])])
#                
#        if to_remove != []:
#            idx_roi = np.delete(idx_roi, to_remove.astype('int'))
#        
#        print(idx_roi)
        
    if direction == 'f':
        for param in parameter_names:
            if '_mu' not in param:  # don't transform orientation parameters
                # NB. Add/subtract 1e-5 to avoid nan/inf if parameter is on upper/lower bound from LS fit. Transformed
                # parameters shouldn't ever reach bounds (i.e. from sampling in Metropolis Hastings step)
                lb = (model.parameter_ranges[param][0] - 1e-5) * model.parameter_scales[param]  # lower bound
                ub = (model.parameter_ranges[param][1] + 1e-5) * model.parameter_scales[param]  # upper bound
                param_dict[param] = np.log(param_dict[param][idx_roi] - lb) - np.log(ub - param_dict[param][idx_roi])

    elif direction == 'r':
        for param in parameter_names:
            if '_mu' not in param:  # don't transform orientation parameters
                # NB. Add/subtract 1e-5 to avoid nan/inf if parameter is on upper/lower bound from LS fit. Transformed
                # parameters shouldn't ever reach bounds (i.e. from sampling in Metropolis Hastings step)
                lb = (model.parameter_ranges[param][0] - 1e-5) * model.parameter_scales[param]  # lower bound
                ub = (model.parameter_ranges[param][1] + 1e-5) * model.parameter_scales[param]  # upper bound
                
                #param_dict[param] = [ub if np.isnan(x) else x for x in param_dict[param]]
                               
                if model.parameter_ranges[param][0] != model.parameter_ranges[param][1]:
                   # tmp = param_dict[param][idx_roi]
                    param_dict[param][idx_roi] = (lb + ub * np.exp(param_dict[param][idx_roi])) / (1 + np.exp(param_dict[param][idx_roi]))
                    if np.any(np.isnan(param_dict[param][idx_roi])):
#                         disp('in tform_params, here0')
                        idx = [x for x in range(param_dict[param][idx_roi].__len__()) if np.isnan(param_dict[param][x])]
                        #idx = find(isnan(param_dict.(param)(idx_roi)));
                        param_dict[param][idx_roi[idx]] = ub
                
                    if np.any(np.isinf(param_dict[param][idx_roi])):
#                         disp('in tform_params, here1')
                        idx = [x for x in range(param_dict[param][idx_roi].__len__()) if np.isinf(param_dict[param][x])]
                        #idx = find(isinf(param_dict.(param)(idx_roi)));
                        param_dict[param][idx_roi[idx]] = lb
                
                else:
                    idx = [x for x in range(param_dict[param][idx_roi].__len__()) if not np.isnan(param_dict[param][x])]
                    #idx = find(~isnan(param_dict.(param)(idx_roi)));
                    param_dict[param][idx_roi[idx]] = lb
                
                if np.any(param_dict[param]==0):
                    print('in tform_params, here2', flush=True)
    else:
        print('Incorrect input! Nothing is happening...', flush=True)

    return param_dict


#==============================================================================
def compute_log_likelihood(model, acq_scheme, signal, predicted_signal, parameter_vector):

    if ( predicted_signal is None and parameter_vector is None ) \
        or ( predicted_signal is not None and parameter_vector is not None ):
        print('Input EITHER "predicted_signal" OR "parameter_struct_new". Exiting...')
        return None, None

    ndw = acq_scheme.bvalues.__len__()

    # actual measured signal
    y = signal

    # model-predicted signal (current params)
    if predicted_signal is not None and parameter_vector is None:
        g = predicted_signal                                                    # use input 
    elif predicted_signal is None and parameter_vector is not None:
        g = model.simulate_signal(acq_scheme, parameter_vector)                 # calculate
    
    # calculate posteriors and PDFs (log scale)
    inner_y_y = np.sum(np.multiply(np.squeeze(y), np.squeeze(y)), 1)
    inner_y_g = np.sum(np.multiply(np.squeeze(y), np.squeeze(g)), 1)
    inner_g_g = np.sum(np.multiply(np.squeeze(g), np.squeeze(g)), 1)
    log_likelihood = (-ndw / 2) * np.log(inner_y_y - (inner_y_g ** 2 / inner_g_g))

    return log_likelihood, g


#==============================================================================
def fit(model, acq_scheme, data, E_fit, params_init_vector, sigma0_norm_struct,
        mu0_norm_struct, mask, nsteps, burn_in, nupdate, update_rois):


    # FIXME: quick hack to allow for different relaxation times in different ROIs 
#    acq_scheme_rois = acq_scheme
#    if length(acq_scheme) > 1
#        acq_scheme = acq_scheme{1};
#    end
    
    # FIXME: data checks?
    print(' >> at start of fit_bayes.fit', flush=True)  # ecap

    # ensure voxel data is in 1D array
    if data.ndim > 2:                                                           # need array size [nvox x ndw]
        data = data.reshape(-1, data.shape[-1])
    if E_fit.ndim > 2:                                                          # need array size [nvox x ndw]
        E_fit = E_fit.reshape(-1, E_fit.shape[-1])       
    if mask is not None and mask.ndim > 1:                                      # need array size [nvox x 1]
        mask = mask.flatten()
    for key in params_init_vector.keys():                                    # need array size [nvox x 1]
        if params_init_vector[key].ndim > 1:
            params_init_vector[key] = params_init_vector[key].flatten()
            
    # set mask default
    if mask is None:
        mask = (data[:, 0] > 0).astype('uint8')
    # convert to int if input as bool
    elif mask.any():
        mask = mask.astype('uint8')

    # scale parameters
    for param in model.parameter_names:
        params_init_vector[param] = params_init_vector[param] * model.parameter_scales[param]
        try:
            params_fixed[param] = params_fixed[param] * model.parameter_scales[param]
        except:
            print('do nothing')
            # do nothing
                

    # extract useful values
    nvox = np.prod(mask.shape)  # np.sum(mask > 0)  number of voxels in mask
    nparams = np.sum(np.array(list(model.parameter_cardinality.values())))
    ndw = len(acq_scheme.bvalues)
    ncomp = model.partial_volume_names.__len__()

    # extract ROIs if present in mask; check enough voxels in each ROI to avoid df error in sigma calculation
    print(' >> extracting ROIs', flush=True)  # ecap
    roi_vals = np.unique(mask)[np.unique(mask) > 0]  # list of unique integers that identify each ROI (ignore 0's)
    roi_nvox = [[xx for xx, x in enumerate(mask == roi_vals[roi]) if x].__len__() for roi in range(roi_vals.__len__())] # number of voxels in each ROI
    to_remove = [roi for roi in range(roi_vals.__len__()) if roi_nvox[roi] < 2 * nparams] # indices of ROIs with too few voxels
    roi_vals = np.delete(roi_vals, to_remove)
    nroi = roi_vals.__len__()  # no. ROIs                                              # no. remaining ROIs

    # get number of compartments; only fit no. compartments - 1
    print(' >> setting up dictionary', flush=True)  # ecap
    model_reduced = deepcopy(model)
    if model.partial_volume_names.__len__() > 0:
        parameters_to_fit = [name for name in model.parameter_names if name != model.partial_volume_names[-1]]
        dependent_fraction = model.partial_volume_names[-1]
        # remove dependent volume fraction from model
        del model_reduced.parameter_ranges[dependent_fraction]
        del model_reduced.parameter_cardinality[dependent_fraction]
        del model_reduced.parameter_scales[dependent_fraction]
        del model_reduced.parameter_types[dependent_fraction]
        del model_reduced.parameter_optimization_flags[dependent_fraction]
    else:
        parameters_to_fit = [name for name in model.parameter_names]
        dependent_fraction = ''

    # remove any parameters that have the same upper and lower bounds
    tmp = model_reduced.parameter_names # because "model_reduced.parameter_names" changes inside loop
    for param in tmp:
        if model_reduced.parameter_ranges[param][0] == model_reduced.parameter_ranges[param][1]:
            del model_reduced.parameter_ranges[param]
            del model_reduced.parameter_cardinality[param]
            del model_reduced.parameter_scales[param]
            del model_reduced.parameter_types[param]
            del model_reduced.parameter_optimization_flags[param]

    parameters_to_fit = model_reduced.parameter_names
    nparams_red = np.sum(np.array(list(model_reduced.parameter_cardinality.values())))  ## ecap edit 20/07/22. nparams -> nparams_red from here on
    
    # create dictionary of model parameter names and LSQ fit values
    params_init = dict.fromkeys(model.parameter_names)
    for param in model.parameter_names:
        params_init[param]= np.squeeze(np.nan * np.ones([nvox, model.parameter_cardinality[param]]))
        params_init[param][mask>0, ] = np.squeeze(params_init_vector[param][mask>0])

    #-------------------------------------------------------------#
    #                     TESTING 18/09/23                        #
#     test_voxel = 911; test_adc = 2.2e-9; test_sigma = .2;
#     params_init.adc(test_voxel) = test_adc;
#     params_init.sigma(test_voxel) = test_sigma;
    #-------------------------------------------------------------#

    params_current = deepcopy(params_init)
    # log transform variables (non-orientation only) (original -> log)
    params_current_tform = tform_params(params_init, model.parameter_names, model, None, 'f'); # dict of initial LSQ fit values
#    parameter_ranges_tform = model.parameter_ranges
#    for param in model.parameter_names:
#        parameter_ranges_tform[param] = (np.array(parameter_ranges_tform[param])) * model.parameter_scales[param]
#    parameter_ranges_tform = tform_params(parameter_ranges_tform, model.parameter_names, model, None, 'f')

    # TODO: play with weights created from ranges - affects convergence
    # initial weights for Metropolis-Hastings parameter sampling
    print(' >> setting up weights', flush=True)
    w = dict.fromkeys(parameters_to_fit)
    for param in parameters_to_fit:  # get scaled parameter ranges
        w[param] = (np.array(model.parameter_ranges[param])) * model.parameter_scales[param]
    w = tform_params(w, parameters_to_fit, model, None, 'f')                    # transform parameter ranges
    w_max = deepcopy(w)                                                                   # create max weight to be used in weight update step
    for param in parameters_to_fit:                                             # set weight as x * range
        if model.parameter_cardinality[param] > 1:
            for card in range(model.parameter_cardinality[param]):
                w[param][card] = 0.001 * np.abs(np.subtract(w[param][card, 1], w[param][card, 0]))
            w[param] = w[param][range(model.parameter_cardinality[param]), 0]
            w[param] = np.tile(w[param], (nvox, 1))                         # tile to create weight for each voxel
        elif model.parameter_cardinality[param] == 1 and param != 'partial_volume_0':
            w[param] = 0.001 * np.abs(np.subtract(w[param][1], w[param][0]))    # 0.001, 0.01, 0.05, 0.1
            w_max[param] = 0.25 * np.abs(np.subtract(w_max[param][1], w_max[param][0]))
            w[param] = np.tile(w[param], nvox)                                  # tile to create weight for each voxel
            w_max[param] = np.tile(w_max[param], nvox)                          # tile to create weight for each voxel
        elif model.parameter_cardinality[param] == 1 and param == 'partial_volume_0':
            w[param] = 0.001 * np.abs(np.subtract(w[param][1], w[param][0]))    # 0.001, 0.01, 0.05, 0.1
            w_max[param] = 0.25 * np.abs(np.subtract(w_max[param][1], w_max[param][0]))
            #w[param] = np.tile(w[param], nvox)                                  # tile to create weight for each voxel
            #w_max[param] = np.tile(w_max[param], nvox)                          # tile to create weight for each voxel

    # initialise variables to track state of optimisation at each step (each parameter tracked independently)
    # FIXEME 08/09/2023: do this when returning variables, not before (get
    # better estimate of posterior if not subsampled before taking mean)
    maxsize = 1e8                                                               # limit max size of arrays (which is otherwise max nsteps*nvox)
    maxsteps = int(np.floor(maxsize/nvox))                                      # max no. updates to store
    whentostore = int(np.ceil(nsteps/maxsteps))                                 # ceil - undershoot max array size
    maxsteps = int(np.floor(nsteps/whentostore))                                # need to recalculate because of rounding in previous line

    accepted = np.squeeze(np.zeros(nvox, dtype=np.float32))                     # total accepted moves over all steps
    accepted_per_n = np.squeeze(np.zeros(nvox, dtype=np.float32))               # accepted moves per N steps (updates weights)
    acceptance_rate = np.squeeze(np.zeros((nvox, nsteps), dtype=np.float32))    # accepted moves at each step

    param_conv = dict.fromkeys(parameters_to_fit)                               # parameter convergence
    gibbs_mu = np.zeros((nroi, nparams_red, maxsteps))                          # gibbs mu at each step (transformed space)
    gibbs_sigma = np.zeros((nroi, nparams_red, nparams_red, maxsteps))          # gibbs sigma at each step (transformed space)
    gibbs_mu_current = np.zeros((nroi, nparams_red))                            # gibbs mu at current step (transformed space) (for ROI membership update)
    gibbs_sigma_current = np.zeros((nroi, nparams_red, nparams_red))            # gibbs sigma at current step (transformed space) (for ROI membership update)
    likelihood_stored = np.squeeze(np.zeros((nvox, maxsteps), dtype=np.float32))# likelihood at each step
    w_stored = dict.fromkeys(parameters_to_fit)                                 # weights at each weight update

    mask_new = np.zeros((nvox,maxsteps), dtype=np.float32)                      # ROI membership at each step
    params_final = dict.fromkeys(model.parameter_names)
    for param in parameters_to_fit:
        if model.parameter_cardinality[param] > 1:
            print('Warning - need to check this cardinality code')
            for card in range(model.parameter_cardinality[param]):
                params_final[param][:, card] = np.zeros(nvox)
        elif model.parameter_cardinality[param] == 1:
            params_final[param] = np.zeros(nvox)

    #-----------------------------------------------------------------------#
    #                                TESTING                                #
    gibbs_mu_norm = np.zeros((nroi, nparams_red, maxsteps), dtype=np.float32)   # gibbs mu at each step (normal space)
    gibbs_sigma_norm = np.zeros((nroi, nparams_red, nparams_red, maxsteps), dtype=np.float32) # gibbs sigma at each step (normal space)
    #-----------------------------------------------------------------------#
    
    # initialise dictionaries (param_conv, accepted, accepted_per_n, acceptance_rate)
    print(' >> no. weight updates = ', int(np.floor(burn_in/nupdate)+1))
    print(' >> initialising dictionaries')
    for param in parameters_to_fit:
        print('    > ', param)
        param_conv[param] = np.squeeze(np.zeros((nvox, model.parameter_cardinality[param], nsteps), dtype=np.float32))
        if nupdate > 0:                                                         # weights update (allow for no updates)
            w_stored[param] = np.squeeze(np.zeros((nvox, model.parameter_cardinality[param], np.int(np.floor(burn_in/(2*nupdate))+1)), dtype=np.float32))  # weights update
        if model.parameter_cardinality[param] > 1:
            for card in range(model.parameter_cardinality[param]):
                w_stored[param][:, card, 0] = w[param][:, card]
        else:
            if nupdate > 0:
                w_stored[param][:, 0] = w[param]

    # calculate priors for hyperprior
    '''
    sigma0_struct = tform_params(sigma0_norm_struct, model.parameter_names, model, None, 'f')  # sigma0 in transformed space
                    tform_params(parameter_ranges_tform, model.parameter_names, model, None, 'f')
    sigma0 = zeros(numel(parameters_to_fit),numel(parameters_to_fit));
    sigma0_norm = zeros(numel(parameters_to_fit),numel(parameters_to_fit));
    for p = 1:numel(parameters_to_fit)
        param = parameters_to_fit{p};
        sigma0(p,p) = sigma0_struct.(param);
        sigma0_norm(p,p) = sigma0_norm_struct.(param);
    end
    mu0_struct = tform_params(mu0_norm_struct, model.parameter_names, model, [], 'f');          # mu0 in transformed space
    mu0 = zeros(1,numel(parameters_to_fit));
    mu0_norm = zeros(1,numel(parameters_to_fit));
    for p = 1:numel(parameters_to_fit)
        param = parameters_to_fit{p};
        mu0(p) = mu0_struct.(param);
        mu0_norm(p) = mu0_norm_struct.(param);
    end
    '''

    #-----------------------------------------------------------------------#
    #------------------------------- MCMC ----------------------------------#
    count = 0
    print(' >> starting loop', flush=True)
    # NB i (voxel loop) and j (MC loop) in keeping with Orton paper
    for j in range(0, nsteps):
        if np.mod(j,100) == 0:
            print('MCMC step = ',j,'/',nsteps)

        # loop over ROIs
        for roi in range(nroi):
#             if mod(j,100) == 0
#                 fprintf('MCMC step = %i/%i; ROI = %i/%i\n', j,nsteps,roi,nroi)
#             end
            idx_roi = [xx for xx, x in enumerate(mask == roi_vals[roi]) if x]   # indices into mask of voxels in ROI
            nvox_roi = idx_roi.__len__()                                        # no. voxels in ROI

            # initialise sigma for this ROI
            if j==0:
                sigma = np.cov(np.transpose(model_reduced.parameters_to_parameter_vector(**params_current_tform)[idx_roi]))

            #-------------------------------------------------------------#
            # TESTING - do calculations in normal parameter ranges
            params_current = tform_params(params_current_tform, model.parameter_names, model, None, 'r')
            params_current_vector = model_reduced.parameters_to_parameter_vector(**params_current)[idx_roi]
            if j==0:
                sigma_norm = np.cov(np.transpose(model_reduced.parameters_to_parameter_vector(**params_current)[idx_roi]))
            #-------------------------------------------------------------#

            # FIXME: quick hack to allow for different relaxation times in different ROIs 
#            if length(acq_scheme_rois) > 1
#                acq_scheme = acq_scheme_rois{roi};
#            end
                
            # Gibbs moves to update priors.
            # Gibbs 1. sample mu from multivariate normal dist defined by current param estimates.
            # Add prior to hyperprior if defined
            params_current_tform_vector = model_reduced.parameters_to_parameter_vector(**params_current_tform)[idx_roi, :]
            m = np.mean(params_current_tform_vector, axis=0)
#            if isempty(sigma0)
            V = sigma / nvox_roi
#            else
#                V = inv( nansum( cat(4,inv(sigma0),nvox_roi*inv(sigma)), 4) );
#            end
#            if ~isempty(mu0)
#                m = (V * nansum( cat(1,mu0*inv(sigma0),nvox_roi*m*inv(sigma)), 1)' )';
#            end
            mu = np.random.multivariate_normal(m, V)

            # Gibbs 2. sample sigma from inverse Wishart distribution (using newly updated mu)
            phi = np.sum([np.outer(params_current_tform_vector[i, :] - mu, params_current_tform_vector[i, :] - mu)
                          for i in range(0, nvox_roi)], axis=0)
            sigma = scipy.stats.invwishart(df=nvox_roi - nparams_red - 1, scale=phi).rvs()
            
            # save Gibbs parameters for this step (careful of parameter ordering)
            if j == 0:
                gibbs_mu[roi,:,j] = copy(mu)
                gibbs_sigma[roi,:,:,j] = copy(sigma)
            elif np.mod(j,whentostore) == 0:
                gibbs_mu[roi,:,int((j/whentostore))] = copy(mu)
                gibbs_sigma[roi,:,:,int((j/whentostore))] = copy(sigma)
            # save tmp version, needed for ROI membership update step
            gibbs_mu_current[roi,:] = copy(mu)
            gibbs_sigma_current[roi,:,:] = copy(sigma)
            
            #-------------------------------------------------------------#
            # TESTING - do calculations in normal parameter ranges
            m_norm = np.mean(params_current_vector, axis=0)
#            if isempty(sigma0)
            V_norm = sigma_norm / nvox_roi
#            else
#                V_norm = inv( nansum( cat(4,inv(sigma0_norm),nvox_roi*inv(sigma_norm)), 4) );
#            end
#            if ~isempty(mu0_norm)
#                m_norm = (V_norm * nansum( cat(1,mu0_norm*inv(sigma0_norm),nvox_roi*m_norm*inv(sigma_norm)), 1)' )';
#            end
            mu_norm = np.random.multivariate_normal(m_norm, V_norm)

            # Gibbs 2. sample sigma from inverse Wishart distribution (using newly updated mu)
            phi_norm = np.sum([np.outer(params_current_vector[i, :] - mu_norm, params_current_vector[i, :] - mu_norm)
                          for i in range(0, nvox_roi)], axis=0)
            sigma_norm = scipy.stats.invwishart(df=nvox_roi - nparams_red - 1, scale=phi_norm).rvs()

            # save Gibbs parameters for this step (careful of parameter ordering)
            if j == 0:
                gibbs_mu_norm[roi,:,j] = copy(mu_norm)
                gibbs_sigma_norm[roi,:,:,j] = copy(sigma_norm)
            elif np.mod(j,whentostore) == 0:
                gibbs_mu_norm[roi,:,int((j/whentostore))] = copy(mu_norm)
                gibbs_sigma_norm[roi,:,:,int((j/whentostore))] = copy(sigma_norm)
            #-------------------------------------------------------------#
            # Metropolis-Hastings parameter updates
            params_new_tform = deepcopy(params_current_tform)

            # UPDATE: all parameters updated at the same time
            for param in parameters_to_fit:
#                 disp(param)
                # sample parameter
                if model.parameter_cardinality[param] > 1:
                    for card in range(model.parameter_cardinality[param]):
                        params_new_tform[param][idx_roi, card] = np.random.normal(params_current_tform[param][idx_roi, card],
                                                                                  w[param][idx_roi, card])
                elif model.parameter_cardinality[param] == 1:
                    params_new_tform[param][idx_roi] = np.random.normal(params_current_tform[param][idx_roi],
                                                                        np.matrix.flatten(w[param][idx_roi]))
                '''
                # FIXME : NEED TO ADD THIS (12 Mar 2024)
                # if a volume fraction was sampled, re-compute dependent fraction
                # create boolean prior to avoid sum(volume fractions) > 1
                if 'partial_volume_' in p and dependent_fraction != '':
                    f_indep = [params_all_new[name][idx_roi] for name in model.partial_volume_names
                               if name != dependent_fraction]
                    f_indep = np.exp(f_indep) / (1 + np.exp(f_indep))  # tform indept fractions (log -> orig)
                    for c in range(ncomp - 1):  # check transform
                        f_indep[c, :] = [1 if np.isnan(x) else x for x in f_indep[c, :]]
                    # prior_new = 1 * (np.sum(f_indep, axis=0) < 1)  # boolean prior to control total vol frac
                    prior_new = np.log(1 * (np.sum(f_indep, axis=0) < 1))  # boolean prior to control total vol frac; 'logged' 10/02/23
                    f_dept = np.array([np.max([0, 1 - np.sum(f_indep[:, f], axis=0)])
                                       for f in range(nvox_roi)])  # compute dept fraction
                    f_dept = np.log(f_dept) - np.log(1 - f_dept)  # tform dept fraction (orig -> log)
                    params_all_new[dependent_fraction][idx_roi] = f_dept
                else:
                    prior_new = np.log(np.ones(nvox_roi))  # dummy prior otherwise
                '''

            #-------------------------------------------------------------#
            #                     TESTING 18/09/23                        #
#             tmp = struct; tmp.adc = test_adc; tmp.sigma = test_sigma; tmp.axr = NaN;
#             tmp = tform_params(tmp, model.parameter_names, model, [], 'f');
#             params_new_tform.adc(test_voxel) = tmp.adc;
#             params_new_tform.sigma(test_voxel) = tmp.sigma;
            #-------------------------------------------------------------#
            
            # compute acceptance
            params_new = tform_params(params_new_tform, model.parameter_names, model, None, 'r'); # transform parameters to normal 
            likelihood, g = compute_log_likelihood(model, acq_scheme, data[idx_roi, :], None,  model_reduced.parameters_to_parameter_vector(**params_current)[idx_roi, :])
            likelihood_new, g_new  = compute_log_likelihood(model, acq_scheme, data[idx_roi, :], None,  model_reduced.parameters_to_parameter_vector(**params_new)[idx_roi, :])

            params_current_tform_vector = model_reduced.parameters_to_parameter_vector(**params_current_tform)[idx_roi, :]
            prior = np.log(scipy.stats.multivariate_normal.pdf(params_current_tform_vector, mu, sigma, allow_singular=1))
            params_new_tform_vector = model_reduced.parameters_to_parameter_vector(**params_new_tform)[idx_roi, :]
#             prior_new = prior_new + log(mvnpdf(parameter_vector, mu, sigma)); # FIXME 
            prior_new = np.log(scipy.stats.multivariate_normal.pdf(params_new_tform_vector, mu, sigma, allow_singular=1))

            # TODO: investigate big discrepancy between r and alpha
            alpha = [(likelihood_new[i] + prior_new[i]) - (likelihood[i] + prior[i]) for i in range(nvox_roi)]
            r = np.log(np.random.uniform(0, 1, nvox_roi))
            #-------------------------------------------------------------#

            # accept new parameter value if criteria met (col 1 -> roi voxel indices, col 2 -> fov voxel indices)
            to_accept = np.array([[i, idx_roi[i]] for i in range(nvox_roi) if r[i] < alpha[i]])
            to_reject = np.array([[i, idx_roi[i]] for i in range(nvox_roi) if r[i] > alpha[i]])

            if to_accept.shape != (0,):  # account for error thrown by no accepted moves
                accepted[to_accept[:, 1]] += 1
                accepted_per_n[to_accept[:, 1]] += 1
                if j == 0:
                    likelihood_stored[to_accept[:, 1], j] = likelihood_new[to_accept[:, 0]] + prior_new[to_accept[:, 0]]
                elif np.mod(j,whentostore) == 0:
                    #-------------------------------------------------------------#
                    #                     TESTING 18/09/23                        #
                    likelihood_stored[to_accept[:, 1], int((j/whentostore))] = likelihood_new[to_accept[:, 0]] + prior_new[to_accept[:, 0]]
#                     likelihood_stored(to_accept(:,2),int((j/whentostore))) = likelihood_new(to_accept(:,1));
                    #-------------------------------------------------------------#

            if to_reject.shape != (0,):  # account for error thrown by all moves accepted
                if j == 0:
                    likelihood_stored[to_reject[:, 1], j] = likelihood[to_reject[:, 0]]
                elif np.mod(j,whentostore) == 0:
                    likelihood_stored[to_reject[:, 1], int((j/whentostore))] = likelihood[to_reject[:, 0]]

            if j == 0:
                acceptance_rate[idx_roi, j] = accepted[idx_roi] / (j+1)
            elif np.mod(j,whentostore) == 0:
                acceptance_rate[idx_roi, int((j/whentostore))] = accepted[idx_roi] / (j+1)

            # update current parameters (based on accepted / rejected)
            for param in parameters_to_fit:
                if model.parameter_cardinality[param] > 1:
                    print('Warning - need to check this cardinality code')
                    for card in range(model.parameter_cardinality[param]):
                        if to_accept.shape != (0,):  # account for error thrown by no accepted moves
                            params_current_tform[param][to_accept[:, 1], card] = copy(params_new_tform[param][to_accept[:, 1], card])
                        if j == 0:
                            param_conv[param][idx_roi, card, j] = tform_params(params_current_tform, model.parameter_names, model, 'r')[param][idx_roi, card]
                        elif np.mod(j,whentostore) == 0:
                            param_conv[param][idx_roi, card, (j/whentostore)] = tform_params(params_current_tform, model.parameter_names, model, 'r')[param][idx_roi, card]
                elif model.parameter_cardinality[param] == 1:
                    if to_accept.shape != (0,):  # account for error thrown by no accepted moves
                        params_current_tform[param][to_accept[:, 1]] = copy(params_new_tform[param][to_accept[:, 1]])

                    if j == 0:
                        param_conv[param][idx_roi, j] = np.array(tform_params(params_current_tform, model.parameter_names, model, None, 'r')[param])[idx_roi]
                    elif np.mod(j,whentostore) == 0:
#                         param_conv.(param)(idx_roi,int((j/whentostore))) = parameters_to_parameter_vector(model_reduced, tmp2, idx_roi);
                        param_conv[param][idx_roi, int((j/whentostore))] = np.array(tform_params(params_current_tform, model.parameter_names, model, None, 'r')[param])[idx_roi]

            # update weights
            if np.mod(j,nupdate) == 0 and 0 < j and j <= burn_in / 2:
                # hack to avoid weight=inf/0 if none/all moves in the last nupdate steps kept
                accepted_per_n[idx_roi] = np.minimum(accepted_per_n[idx_roi], nupdate)
                for param in parameters_to_fit:
                    if model.parameter_cardinality[param] > 1:
                         for card in range(model.parameter_cardinality[param]):
                             print('Warning - need to check this cardinality code')
                             w[param][idx_roi, card] = np.minimum( w_max[param][idx_roi,card],  (w[param][idx_roi, card] * (nupdate + 1)) / ((1 / (1 - .25)) * ((nupdate + 1) - accepted_per_n[idx_roi]/model.parameter_cardinality[param])))
                             w_stored[param][idx_roi, card, np.int((j+1)/nupdate)] = w[param][idx_roi, card]
                    elif model.parameter_cardinality[param] == 1:
                        w[param][idx_roi] = np.minimum( w_max[param][idx_roi], (w[param][idx_roi] * (nupdate + 1)) / ((1/(1-.25)) * ((nupdate+1)-accepted_per_n[idx_roi])) )
                        w_stored[param][idx_roi, np.int((j+1)/nupdate)] = w[param][idx_roi]
                accepted_per_n[idx_roi] = np.zeros((nvox_roi,))
        # end of roi loop
        
        # do ROI membership update
        '''
        if update_rois
            # latest parameter vector
            params_current_tform_vector_all = parameters_to_parameter_vector(model_reduced, params_current_tform, 1:nvox);
    
            # calculate prior PDFs (log scale) of each of voxel of being in any/all rois
            posterior_for_rois = zeros(nvox,nroi); # [nvox, nparams]
            for roi = 1:nroi
#                 posterior_for_rois(:, roi) = logmvnpdf(params_current_tform_vector_all, gibbs_mu(roi, :, j), squeeze(gibbs_sigma(roi, :, :, j)))';
                posterior_for_rois(:, roi) = logmvnpdf(params_current_tform_vector_all, gibbs_mu_current(roi,:), squeeze(gibbs_sigma_current(roi,:,:)))';
            end

            # normalise prior probs
#             posterior_for_rois = exp(posterior_for_rois) ./ (repmat(sum(exp(posterior_for_rois),2),[1,2])); # rows should sum to 1    
            TheTrick = max(posterior_for_rois,[],2) + log(sum(exp(posterior_for_rois-max(posterior_for_rois,[],2)),2)); # to avoid underflow
            posterior_for_rois = posterior_for_rois - TheTrick;
#             posterior_for_rois = posterior_for_rois - repmat(log(sum(exp(posterior_for_rois),2)),[1,2]); 
            
            if j == 1
                mask_new(:,j) = mask;
            end
            
            idx_mask = find(logical(mask));
            mask(idx_mask) = arrayfun(@(ii) randsample(nroi,1,true,exp(posterior_for_rois(ii,:))), idx_mask)';
            
            if mod(j,whentostore) == 0
                mask_new(:,int((j/whentostore))) = mask;
            end
        end
        '''
        
        # sum parameter estimates (divide by burn_in after mcmc loop) (saves memory - avoids saving every step individually) 
        if j > burn_in:
            params_current = tform_params(params_current_tform, model.parameter_names, model, None, 'r')
            count += 1
            for param in parameters_to_fit:
                if model.parameter_cardinality[param] > 1:
                    print('Warning - need to check this cardinality code')
                    for card in range(model.parameter_cardinality[param]):
                        params_final[param][:, card] = params_final[param][:, card] + params_current[param][:, card]
                elif model.parameter_cardinality[param] == 1:
                    params_final[param] = params_final[param] + params_current[param]

    # end of MC step loop

    # get mean of parameter estimates (divide sum by burn_in)
    for param in parameters_to_fit:
        if model.parameter_cardinality[param] > 1:
            print('Warning - need to check this cardinality code')
            for card in range(model.parameter_cardinality[param]):
                params_final[param][:, card] = params_final[param][:, card] / (nsteps-burn_in)
        elif model.parameter_cardinality[param] == 1:
            params_final[param] = params_final[param] / (nsteps-burn_in)

#    for p = 1:numel(parameters_to_fit)
#        param = parameters_to_fit{p};
#        if model.parameter_cardinality.(param) > 1
#            error('Code conversion still needed re. parameter cardinality > 1 (5). Exiting...')
##             for card in range(model.parameter_cardinality[param]):
##                 params_all[param][:, card] = np.mean(param_conv[param][:, card, burn_in:-1], axis=1)
#        elseif model.parameter_cardinality.(param) == 1
#            params_final.(param) = mean(param_conv.(param)(:,ceil(burn_in/whentostore):end),2);
#        end
#    end

    # add back the fixed parameters / dependent volume fractions
    toaddback = [x for x in model.parameter_names if x not in parameters_to_fit]
    for param in toaddback:
        print('Warning - need to check this code')
        params_final[param] = params_init[param]

    return params_final, acceptance_rate, param_conv, likelihood_stored, w_stored, gibbs_mu, gibbs_sigma, gibbs_mu_norm, gibbs_sigma_norm, mask_new
