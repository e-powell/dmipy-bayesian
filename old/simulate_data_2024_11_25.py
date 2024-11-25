#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 2 09:09:09 2022

@author: epowell
"""
import numpy as np
import math
import shelve
from contextlib import contextmanager
import sys
import os

from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues


# simulate signals using SMT NODDI model
def _smt_noddi(smt_noddi_model, acq_scheme):

    # number of voxels for each tissue type
    n_wm = 500
    n_gm = 300
    n_csf = 100
    
    # limits on the parameters for wm/gm/csf
    stick_par_wm = [1e-9, 2e-9]
    stick_par_gm = [1e-9, 2e-9]
    stick_par_csf = [1e-9, 2e-9]
    zep_par_wm = [1.5e-9, 2.5e-9]
    zep_par_gm = [1.5e-9, 2.5e-9]
    zep_par_csf = [1.5e-9, 2.5e-9]
    odi_wm = [0.01, 0.3]
    odi_gm = [0.6, 0.9]
    odi_csf = [0, 1]
    f_stick_wm = [0.7, 0.9]
    f_stick_gm = [0.6, 8]
    f_stick_csf = [0.8, 1]
    f_bundle_wm = [0.7, 0.9]
    f_bundle_gm = [0.8, 1]
    f_bundle_csf = [0, 0.2]


    # roi_mask = np.concatenate((np.ones((n_wm,)),2*np.ones((n_gm,)),3*np.ones((n_csf,))))
    # TODO: change to normal dist
    roi_mask = np.concatenate((np.ones((n_wm,)),2*np.ones((n_gm,)),np.zeros((n_csf,))))

    stick_par = np.concatenate((np.random.uniform(low=stick_par_wm[0],high=stick_par_wm[1], size = n_wm),
                          np.random.uniform(low=stick_par_gm[0],high=stick_par_gm[1], size = n_gm),
                          np.random.uniform(low=stick_par_csf[0],high=stick_par_csf[1], size = n_csf)))

    zep_par = np.concatenate((np.random.uniform(low=zep_par_wm[0],high=zep_par_wm[1], size = n_wm),
                          np.random.uniform(low=zep_par_gm[0],high=zep_par_gm[1], size = n_gm),
                          np.random.uniform(low=zep_par_csf[0],high=zep_par_csf[1], size = n_csf)))

    odi = np.concatenate((np.random.uniform(low=odi_wm[0],high=odi_wm[1], size = n_wm),
                          np.random.uniform(low=odi_gm[0],high=odi_gm[1], size = n_gm),
                          np.random.uniform(low=odi_csf[0],high=odi_csf[1], size = n_csf)))                

    f_stick = np.concatenate((np.random.uniform(low=f_bundle_wm[0],high=f_bundle_wm[1], size = n_wm),
                          np.random.uniform(low=f_bundle_gm[0],high=f_bundle_gm[1], size = n_gm),
                          np.random.uniform(low=f_bundle_csf[0],high=f_bundle_csf[1], size = n_csf)))

    f_bundle = np.concatenate((np.random.uniform(low=f_bundle_wm[0],high=f_bundle_wm[1], size = n_wm),
                          np.random.uniform(low=f_bundle_gm[0],high=f_bundle_gm[1], size = n_gm),
                          np.random.uniform(low=f_bundle_csf[0],high=f_bundle_csf[1], size = n_csf)))

    f_free = 1 - f_bundle

    # put into a big parameter vector that can be passed into simulate_signal
    parameter_vector = smt_noddi_model.parameters_to_parameter_vector(BundleModel_1_partial_volume_0=f_stick,
                                                                    partial_volume_0=f_bundle,
                                                                    partial_volume_1=f_free)
    
    signals = smt_noddi_model.simulate_signal(acq_scheme, parameter_vector)
    
    return signals, parameter_vector, roi_mask


# simulate signals using stick-zeppelin model (directional)
def _sz_directional(sz_model, acq_scheme):

    # number of voxels for each tissue type
    n_wm = 500
    n_gm = 300
    n_csf = 100
    
    # limits on the parameters for wm/gm/csf
    lambda_par_wm = [1.5e-9, 2.5e-9]
    lambda_par_gm = [0.5e-9, 1.5e-9]
    lambda_par_csf = [2.9e-9, 3e-9]
    mu_1 = [0, np.pi]
    mu_2 = [-np.pi,np.pi]
    f_stick_wm = [0.6, 0.8]
    f_stick_gm = [0.2, 0.4]
    f_stick_csf = [0, 0.01]
    
    
    roi_mask = np.concatenate((np.ones((n_wm,)),2*np.ones((n_gm,)),3*np.ones((n_csf,))))                      

    lambda_par = np.concatenate((np.random.uniform(low=lambda_par_wm[0],high=lambda_par_wm[1], size = n_wm),
                          np.random.uniform(low=lambda_par_gm[0],high=lambda_par_gm[1], size = n_gm),
                          np.random.uniform(low=lambda_par_csf[0],high=lambda_par_csf[1], size = n_csf)))
                  
    f_stick = np.concatenate((np.random.uniform(low=f_stick_wm[0],high=f_stick_wm[1], size = n_wm),
                          np.random.uniform(low=f_stick_gm[0],high=f_stick_gm[1], size = n_gm),
                          np.random.uniform(low=f_stick_csf[0],high=f_stick_csf[1], size = n_csf)))
    
    mu_1 = np.random.uniform(low=mu_1[0],high=mu_1[1],size=n_wm+n_gm+n_csf)
    mu_2 = np.random.uniform(low=mu_2[0],high=mu_2[1],size=n_wm+n_gm+n_csf)
    
    mu = np.stack((mu_1,mu_2),axis=1)
    
    f_free = 1 - f_stick
    
    # put into a big parameter vector that can be passed into simulate_signal
    parameter_vector = sz_model.parameters_to_parameter_vector(BundleModel_1_G2Zeppelin_1_mu=mu,
                                                                    BundleModel_1_G2Zeppelin_1_lambda_par=lambda_par,
                                                                    BundleModel_1_partial_volume_0=f_stick)
    
    signals = sz_model.simulate_signal(acq_scheme, parameter_vector)
    
    return signals, parameter_vector, roi_mask


