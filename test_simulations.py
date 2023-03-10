#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 2 09:09:09 2022

@author: epowell
"""
# %% load some necessary modules

import numpy as np
from copy import copy, deepcopy
import time
from importlib import reload
import matplotlib.pyplot as plt
import random

# load HCP acqusition scheme
from dmipy.data import saved_acquisition_schemes
from dmipy.utils import spherical_mean

import fit_bayes_new 
fit_bayes_new = reload(fit_bayes_new)

from useful_functions import suppress_stdout, compute_time, create_spherical_mean_scheme
from useful_functions import add_noise, check_lsq_fit, mask_from_tensor_model, make_square_axes
import setup_models, simulate_data


# %% setup acquisition scheme
acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
acq_scheme_smt = create_spherical_mean_scheme(acq_scheme)
    
# %% set up models
# smt_noddi = setup_models._smt_noddi()   # SMT NODDI
sz = setup_models._sz()              # directional stick-zeppelin
sz_sm = setup_models._sz_sm()           # spherical mean stick-zeppelin
    
# %% generate ground truth parameters and corresponding (directional) signals
# signals_gt, parameters_smt_noddi, mask = simulate_data(smt_noddi, acq_scheme)
signals_gt, parameter_vect_gt, roi_mask_gt = simulate_data._sz_directional(sz, acq_scheme)
parameter_dict_gt = sz.parameter_vector_to_parameters(parameter_vect_gt)
# add noise
signals_noisy = add_noise(signals_gt, snr=10)

fig, ax = plt.subplots(1, 2)
ax[0].plot(acq_scheme.bvalues, signals_gt[0,:],'o', color='seagreen')
ax[1].plot(acq_scheme.bvalues,signals_noisy[0,:],'o', color='steelblue')

# %% get roi mask from DT fit to noisy data
roi_mask = mask_from_tensor_model(signals_noisy, acq_scheme)

plt.plot(roi_mask_gt-roi_mask)

# %% calculate spherical mean of signals
signals_gt_sm = np.asarray([spherical_mean.estimate_spherical_mean_multi_shell(signals_gt[i,:], acq_scheme) for i in range(0,signals_gt.shape[0])])
signals_noisy_sm = np.asarray([spherical_mean.estimate_spherical_mean_multi_shell(signals_noisy[i,:], acq_scheme) for i in range(0,signals_noisy.shape[0])])

fig, ax = plt.subplots(1, 2)
ax[0].plot(acq_scheme_smt.bvalues, signals_gt_sm[0,:],'o-', color='seagreen')
ax[1].plot(acq_scheme_smt.bvalues, signals_noisy_sm[0,:],'o-', color='steelblue')


# %% LSQ fitting
# lsq_fit = smt_noddi.fit(acq_scheme_smt, signals_snr100)
with suppress_stdout():  # suppress annoying output in console
    lsq_fit = sz.fit(acq_scheme, signals_noisy)                 # directional model
    lsq_fit_sm = sz_sm.fit(acq_scheme, signals_noisy)           # spherical mean model

# check LSQ fits don't hit the bounds; add/subtract eps to any that do
# parameters_lsq_sm_dict = check_lsq_fit(sz_sm, lsq_fit_sm.fitted_parameters)

fig, ax = plt.subplots(1, 3)
for roi in range(0,int(np.max(roi_mask_gt))+1):
    ax[0].plot(parameter_dict_gt['BundleModel_1_G2Zeppelin_1_mu'][roi_mask_gt==roi], lsq_fit.fitted_parameters['BundleModel_1_G2Zeppelin_1_mu'][roi_mask_gt==roi],'o',markersize=1)
    ax[1].plot(parameter_dict_gt['BundleModel_1_G2Zeppelin_1_lambda_par'][roi_mask_gt==roi], lsq_fit.fitted_parameters['BundleModel_1_G2Zeppelin_1_lambda_par'][roi_mask_gt==roi],'o',markersize=1)
    ax[2].plot(parameter_dict_gt['BundleModel_1_partial_volume_0'][roi_mask_gt==roi], lsq_fit.fitted_parameters['BundleModel_1_partial_volume_0'][roi_mask_gt==roi],'o',markersize=1)
    ax[1].set_ylim([0, 3e-9])
    ax[2].set_ylim([0, 1])

fig, ax = plt.subplots(1, 3)
for roi in range(0,int(np.max(roi_mask_gt))+1):
    ax[1].plot(parameter_dict_gt['BundleModel_1_G2Zeppelin_1_lambda_par'][roi_mask_gt==roi], lsq_fit_sm.fitted_parameters['BundleModel_1_G2Zeppelin_1_lambda_par'][roi_mask_gt==roi],'o',markersize=1)
    ax[2].plot(parameter_dict_gt['BundleModel_1_partial_volume_0'][roi_mask_gt==roi], lsq_fit_sm.fitted_parameters['BundleModel_1_partial_volume_0'][roi_mask_gt==roi],'o',markersize=1)
    ax[1].set_ylim([0, 3e-9])
    ax[2].set_ylim([0, 1])

# %% LSQ-predicted signals
with suppress_stdout():  # suppress annoying output in console
    # E_fit = np.squeeze(smt_noddi.simulate_signal(acq_scheme_smt, parameters_lsq_vect))
    E_fit = np.squeeze(sz.simulate_signal(acq_scheme, lsq_fit.fitted_parameters_vector))
    E_fit_sm = np.squeeze(sz_sm.simulate_signal(acq_scheme_smt, lsq_fit_sm.fitted_parameters_vector))

for i in random.sample(range(signals_gt.shape[0]), 10):
    plt.plot(signals_gt[i,:]- E_fit[i,:],'o-')
fig, ax = plt.subplots(1, 2)
for i in random.sample(range(signals_gt.shape[0]), 10):
    ax[0].plot(acq_scheme_smt.bvalues, signals_gt_sm[i,:],'o-')
    ax[1].plot(acq_scheme_smt.bvalues, E_fit_sm[i,:],'o-')


# %% hierarchical Bayesian fitting
nsteps = 2000
burn_in = 1000
nupdates = 20
proc_start = time.time()
parameters_dict_bayes, acceptance_rate, parameter_convergence, likelihood, weights\
    = fit_bayes_new.fit(sz_sm, acq_scheme_smt, signals_noisy_sm, E_fit_sm, lsq_fit_sm.fitted_parameters, roi_mask_gt, nsteps, burn_in, nupdates)
compute_time(proc_start, time.time())

fig, ax = plt.subplots(1, 3)
for roi in range(0,int(np.max(roi_mask_gt))+1):
    ax[1].plot(parameter_dict_gt['BundleModel_1_G2Zeppelin_1_lambda_par'][roi_mask_gt==roi], parameters_dict_bayes['BundleModel_1_G2Zeppelin_1_lambda_par'][roi_mask_gt==roi],'o',markersize=1)
    ax[2].plot(parameter_dict_gt['BundleModel_1_partial_volume_0'][roi_mask_gt==roi], parameters_dict_bayes['BundleModel_1_partial_volume_0'][roi_mask_gt==roi],'o',markersize=1)
    ax[1].set_ylim([0, 3e-9])
    ax[2].set_ylim([0, 1])

fig, ax = plt.subplots(1, 4)
for i in random.sample(range(signals_gt.shape[0]), 10):
    ax[1].plot(parameter_convergence['BundleModel_1_G2Zeppelin_1_lambda_par'][i,:])
    ax[2].plot(parameter_convergence['BundleModel_1_partial_volume_0'][i,:])
    ax[3].plot(acceptance_rate[i,:])
