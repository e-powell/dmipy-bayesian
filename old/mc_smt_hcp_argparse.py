#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:37:36 2023

@author: epowell
"""

# %% load some necessary modules

import numpy as np
from copy import copy, deepcopy
import time
from importlib import reload
import matplotlib.pyplot as plt
import random
import nibabel as nib
import pickle
import argparse

# load HCP acqusition scheme
from dmipy.data import saved_acquisition_schemes
from dmipy.utils import spherical_mean

import fit_bayes_new 
fit_bayes_new = reload(fit_bayes_new)

from useful_functions import suppress_stdout, compute_time, create_spherical_mean_scheme
from useful_functions import add_noise, check_lsq_fit, mask_from_kmeans, make_square_axes, mask_from_gmm
import setup_models

# To run: python mc_smt_hcp_argparse.py --nsteps=200 --burn_in=60 --nupdates=20 --path_dmri='/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/data.nii.gz' --path_mask='/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/nodif_brain_mask.nii.gz' --path_lsqfit_nii='/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/lsq_fit.nii.gz' --path_bayfit_nii='/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/bayes_fit.nii.gz' --slice=70

def main():
    print('We are in!', flush=True)
    t_total = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsteps', type=int, nargs='+', help='Number of MCMC steps')
    parser.add_argument('--burn_in', type=int, nargs='+', help='Number of burn in steps for MCMC')
    parser.add_argument('--nupdates', type=int, nargs='+', help='How often to update jumping weights in MCMC (every nupdates steps)')
    parser.add_argument('--path_dmri', type=str, nargs='+', help='Full path to dMRI data')
    parser.add_argument('--path_mask', type=str, nargs='+', help='Full path to mask (either global or tissue-specific)')
    parser.add_argument('--path_dmri_sm', type=str, nargs='+', help='Full path to output location for savigng spherical mean of dMRI data')
    parser.add_argument('--path_lsqfit_nii', type=str, nargs='+', help='Full path to output location for saving LSQ fits')
    parser.add_argument('--path_bayfit_nii', type=str, nargs='+', help='Full path to output location for saving Bayesian fits')
    parser.add_argument('--path_rois_nii', type=str, nargs='+', help='Full path to output location for saving ROIs from k-means clustering')
    parser.add_argument('--slice', type=int, nargs='+', help='Which slice(s) to fit data to (no argument required for full FoV)')
    args = parser.parse_args()
    print(args, flush=True)
    nsteps = args.nsteps[0]
    burn_in = args.burn_in[0]
    nupdates = args.nupdates[0]
    path_dmri = args.path_dmri[0]
    path_mask = args.path_mask[0]
    path_dmri_sm = args.path_dmri_sm[0]
    path_lsqfit_nii = args.path_lsqfit_nii[0]
    path_bayfit_nii = args.path_bayfit_nii[0]
    path_rois_nii = args.path_rois_nii[0]
    # path_bval = args.path_bval[0]
    # path_bvec = args.path_bvec[0]
    # path_amico = args.path_amico[0]
    # hdr = nib.load(path_mask)
    # hdr = hdr.header
    if args.slice is not None:
        slice = args.slice[0]
    else:
        slice = np.nan

    print(str(slice))
    
    # %% load HCP test and retest data 
    print('Loading data and mask...', flush=True)
    data = nib.load(path_dmri).get_fdata(dtype=np.float32)
    mask = nib.load(path_mask).get_fdata(dtype=np.float32)  # just use to speed up LSQ fitting
    print('Done.', flush=True)
    
    # %% setup acquisition scheme
    acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    acq_scheme_smt = create_spherical_mean_scheme(acq_scheme)
    # slice = 70
    n_tissue = 2
    
    # %% set up models
    # smt_noddi = setup_models._smt_noddi()   # SMT NODDI
    sz = setup_models._sz()                 # directional stick-zeppelin
    sz_sm = setup_models._sz_sm()           # spherical mean stick-zeppelin
    n_params = sz_sm.parameter_names.__len__()
    n_bval = acq_scheme_smt.bvalues.__len__()
    
    # %% reshape data
    if not np.isnan(slice):
        nz_orig = data.shape[2]
        data = data[:,:,slice,:]
        mask = mask[:,:,slice]
    
    nx = data.shape[0]
    ny = data.shape[1]
    if data.ndim == 3:
        nz = 1
        ndw = data.shape[2]
        data = data * np.tile(mask[:,:,np.newaxis],(1,1,data.shape[-1]))        # apply mask to get rid of extraneous voxels
        scale = np.ones((nx,ny,nz_orig,n_params))
    elif data.ndim == 4:
        nz = data.shape[2]
        ndw = data.shape[3]
        data = data * np.tile(mask[:,:,:,np.newaxis],(1,1,1,data.shape[-1]))    # apply mask to get rid of extraneous voxels
        scale = np.ones((nx,ny,nz,n_params))
    
    scale[:,:,:,0] = scale[:,:,:,0]*1e9
    dims = [nx, ny, nz]
    # data = np.reshape(data, [nx*ny*nz,ndw])
    # mask = np.reshape(mask, [nx*ny*nz])

    # %% calculate spherical mean of signals
    print('Calculating spherical mean of data', flush=True)
    t_sm = time.time()
    tmp = data.reshape(-1, data.shape[-1])[mask.flatten()>0,:]
    data_sm = np.zeros((nx*ny*nz,n_bval))
    data_sm[mask.flatten()>0,:] = np.asarray([spherical_mean.estimate_spherical_mean_multi_shell(tmp[i,:], acq_scheme) for i in range(0,tmp.shape[0])])
    data_sm = np.reshape(data_sm, [nx,ny,nz,n_bval])
    print('Saving spherical mean data nifti...', flush=True)
    hdr_data = nib.load(path_mask).header.copy()
    hdr_data['dim'][0:5] = np.insert((nx,ny,nz_orig,n_bval), 0, 4)              # modify header with correct image dimensions
    hdr_data['pixdim'][5] = 1                                                   # add pixdim for 4th dim
    if data.ndim == 3:
        tmp = np.empty((nx,ny,nz_orig,n_bval)) * np.NaN
        tmp[:,:,slice,:] = data_sm.squeeze()
    else:
        tmp = lsq_fit_data.fitted_parameters_vector
    img = nib.Nifti1Image(tmp, None, header=hdr_data)
    img.get_data_dtype()
    nib.save(img, path_dmri_sm) 
    compute_time(t_sm, time.time())
    print('Done.', flush=True)
    
    # %% run LSQ fitting and save
    print('Running LSQ fit...', flush=True)
    t_lsq = time.time()
    lsq_fit_data = sz_sm.fit(acq_scheme, data, mask, use_parallel_processing=False)
    # save nii
    print('Saving LSQ fit as nifti...', flush=True)
    hdr_lsq = nib.load(path_mask).header.copy()
    hdr_lsq['dim'][0:5] = np.insert((nx,ny,nz_orig,n_params), 0, 4)              # modify header with correct image dimensions
    hdr_lsq['pixdim'][5] = 1                                                    # add pixdim for 4th dim
    if data.ndim == 3:
        tmp = np.empty((nx,ny,nz_orig,n_params)) * np.NaN
        tmp[:,:,slice,:] = lsq_fit_data.fitted_parameters_vector
    else:
        tmp = lsq_fit_data.fitted_parameters_vector
    img = np.multiply(scale, tmp)                                               # array, scale so parameters of same order (easier for viewing)
    img = nib.Nifti1Image(img, None, header=hdr_lsq)                            # convert array to nifti structure
    img.get_data_dtype()
    nib.save(img, path_lsqfit_nii)
    compute_time(t_lsq, time.time())
    print('Done.', flush=True)
    # '''
    
    # %% calculate LSQ-predicted signals
    print('Calculating signals from LSQ fit...', flush=True)
    E_fit_sm = np.squeeze(sz_sm.simulate_signal(acq_scheme, lsq_fit_data.fitted_parameters_vector))
    print('Done.', flush=True)
    
    # %% Mask out CSF (diffusivity > 2.95e-9; stick fraction < 0.05)
    tmp_data = deepcopy(np.reshape(lsq_fit_data.fitted_parameters_vector,[nx*ny*nz,lsq_fit_data.fitted_parameters_vector.shape[-1]])) #[mask>0,:]
    # tmp_data = tmp_data[np.ndarray.flatten(mask>0),:]
    wmgm_idx = ((tmp_data[:,0]<2.95e-9) & (tmp_data[:,1]>0.05)) | (tmp_data[:,1]>0.5)                   # indices of WM/GM only
    
    # %% get roi mask from k-means / GMM clustering 
    # print('Estimating tissue ROIs from k-means clustering on LSQ fits...', flush=True)
    print('Estimating tissue ROIs from GMM on LSQ fits...', flush=True)
    # tmp_roi = mask_from_kmeans(parameter_vector=tmp_data, n_clusters=n_tissue+1)  # mask from k-means
    tmp_data[:,0] = tmp_data[:,0]*1e9
    tmp_roi, _ = mask_from_gmm(parameter_vector=tmp_data[wmgm_idx,:], n_clusters=n_tissue)  # mask from GMM
    roi_mask = np.zeros([nx*ny*nz])
    roi_mask[wmgm_idx] = tmp_roi
    roi_mask = np.squeeze(np.reshape(roi_mask, [nx,ny,nz]))
    print('Saving as nifti...', flush=True)
    hdr_rois = nib.load(path_mask).header.copy()
    hdr_rois['dim'][0:4] = np.insert((nx,ny,nz_orig), 0, data.ndim)              # modify header with correct image dimensions
    if data.ndim == 3:
        tmp = np.empty((nx,ny,nz_orig)) * np.NaN
        tmp[:,:,slice] = roi_mask
    else:
        tmp = roi_mask
    img = nib.Nifti1Image(tmp, None, header=hdr_rois)
    img.get_data_dtype()
    nib.save(img, path_rois_nii) 
    print('Done', flush=True)
    

    # %% hierarchical Bayesian fitting
    print('Running Bayesian fit...', flush=True)
    t_bayes = time.time()
    parameter_dict = lsq_fit_data.fitted_parameters
    parameters_dict_bayes, acceptance_rate, parameter_convergence, likelihood, weights\
        = fit_bayes_new.fit(model=sz_sm, acq_scheme=acq_scheme, data=data_sm, 
                            E_fit=E_fit_sm, parameter_vector_init=parameter_dict, 
                            mask=roi_mask, nsteps=nsteps, burn_in=burn_in, nupdate=nupdates)
    # parameters_dict_bayes, acceptance_rate, parameter_convergence, likelihood, weights\
        # = fit_bayes_roi_update.fit(model=sz_sm, acq_scheme=acq_scheme, data=data_sm, E_fit=E_fit_sm, parameter_vector_init=parameter_dict, mask=roi_mask, nsteps=nsteps, burn_in=burn_in, nupdates=nupdates)
    compute_time(t_bayes, time.time())

    # %% reshape and save
    print('Saving Bayesian fit...', flush=True)
    parameters_vect_bayes = np.zeros((nx*ny,parameters_dict_bayes.keys().__len__()))
    i=0
    for key in parameters_dict_bayes.keys():
        parameters_vect_bayes[:,i] = parameters_dict_bayes.get(key)
        i+=1
    parameters_vect_bayes = np.reshape(parameters_vect_bayes,[nx,ny,parameters_dict_bayes.keys().__len__()])

    print('Saving as nifti...', flush=True)
    if data.ndim == 3:
        parameters_vect_bayes = parameters_vect_bayes[:,:,np.newaxis,:]
    hdr_bay = nib.load(path_mask).header.copy()
    hdr_bay['dim'][0:5] = np.insert((nx,ny,nz_orig,n_params), 0, data.ndim)              # modify header with correct image dimensions
    hdr_bay['pixdim'][5] = 1                                                    # add pixdim for 4th dim
    if data.ndim == 3:
        tmp = np.empty((nx,ny,nz_orig,n_params)) * np.NaN
        tmp[:,:,slice,:] = parameters_vect_bayes.squeeze()
    else:
        tmp = parameters_vect_bayes
    img = np.multiply(scale, tmp)                                               # array, scale so parameters of same order (easier for viewing)
    img = nib.Nifti1Image(img, None, header=hdr_bay)                            # convert array to nifti structure
    img.get_data_dtype()
    nib.save(img, path_bayfit_nii)
    print('Done.', flush=True)
    # '''
    compute_time(t_total, time.time())

if __name__ == '__main__':
    main()