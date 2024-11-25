#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:27:22 2023

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

# load HCP acqusition scheme
from dmipy.data import saved_acquisition_schemes
from dmipy.utils import spherical_mean

import fit_bayes_new, fit_bayes_roi_update
fit_bayes_new = reload(fit_bayes_new)
fit_bayes_roi_update = reload(fit_bayes_roi_update)

from useful_functions import suppress_stdout, compute_time, create_spherical_mean_scheme
from useful_functions import add_noise, check_lsq_fit, mask_from_kmeans, make_square_axes, mask_from_gmm
import setup_models, simulate_data


# %% setup acquisition scheme
acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
acq_scheme_smt = create_spherical_mean_scheme(acq_scheme)
slice = 70
n_tissue = 2

# %% set up models
# smt_noddi = setup_models._smt_noddi()   # SMT NODDI
sz = setup_models._sz()                 # directional stick-zeppelin
sz_sm = setup_models._sz_sm()           # spherical mean stick-zeppelin
    
# %% load HCP test and retest data   
path_dmri = '/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/data.nii.gz'
path_mask = '/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/nodif_brain_mask.nii.gz'
path_dmri_sm = '/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/data_sm.nii.gz'
path_lsqfit_nii = '/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/lsq_fit.nii.gz'
path_lsqfit_pckl = '/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/sz_sm.pckl'
path_Efit_pckl = '/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/E_fit.pckl'
path_rois_nii = '/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/rois_gmm.nii'
path_bayfit_nii = '/home/epowell/data/hcp/diffusion_retest/103818_2/T1w/Diffusion/bay_fit.nii.gz'
data_orig = nib.load(path_dmri).get_fdata(dtype=np.float32)
mask_orig = nib.load(path_mask).get_fdata(dtype=np.float32)  # just use to speed up LSQ fitting
hdr = nib.load(path_mask).header.copy()

# %% reshape data

# data = data_orig[:,:,slice,:]
# mask = mask_orig[:,:,slice]
data = data_orig
mask = mask_orig
data = data * np.tile(mask[:,:,:,np.newaxis],(1,1,1,data.shape[-1]))

nx = data.shape[0]
ny = data.shape[1]
if data.ndim == 3:
    nz = 1
    ndw = data.shape[2]
elif data.ndim == 4:
    nz = data.shape[2]
    ndw = data.shape[3]

# data = np.reshape(data, [nx*ny*nz,ndw])
# mask = np.reshape(mask, [nx*ny*nz])

# %% calculate spherical mean of signals
'''
tmp = data.reshape(-1, data.shape[-1])[mask.flatten()>0,:]
data_sm = np.zeros((nx*ny*nz,acq_scheme_smt.bvalues.__len__()))
data_sm[mask.flatten()>0,:] = np.asarray([spherical_mean.estimate_spherical_mean_multi_shell(tmp[i,:], acq_scheme) for i in range(0,tmp.shape[0])])
data_sm = np.reshape(data_sm, [nx,ny,nz,acq_scheme_smt.bvalues.__len__()])
# save nii
hdr_data = nib.load(path_mask).header.copy()
hdr_data['dim'][0:5] = np.insert(data_sm.shape, 0, data.ndim)                   # modify header with correct image dimensions
hdr_data['pixdim'][5] = 1                                                       # add pixdim for 4th dim
img = nib.Nifti1Image(data_sm, None, header=hdr_data)
img.get_data_dtype()
nib.save(img, path_dmri_sm) 
'''
# load nii
data_sm = nib.load(path_dmri_sm).get_fdata()


# %% run LSQ fitting
'''
lsq_fit_data = sz_sm.fit(acq_scheme, data, mask)
# save nii
hdr_lsq = nib.load(path_mask).header.copy()
hdr_lsq['dim'][0:5] = np.insert(lsq_fit_data.fitted_parameters_vector.shape, 0, data.ndim) # modify header with correct image dimensions
hdr_lsq['pixdim'][5] = 1                                                        # add pixdim for 4th dim
scale = np.ones_like(lsq_fit_data.fitted_parameters_vector)
scale[:,:,:,0] = scale[:,:,:,0]*1e9
img = np.multiply(scale, lsq_fit_data.fitted_parameters_vector)                 # array, scale so parameters of same order (easier for viewing)
img = nib.Nifti1Image(img, None, header=hdr_lsq)                                # covnert array to nifti structure
img.get_data_dtype()
nib.save(img, path_lsqfit_nii)
compute_time(t_lsq, time.time())
# load nii
img = nib.load(path_lsqfit_nii)
img = np.multiply(scale, img)
# pickle
f = open(path_lsqfit_pckl, 'wb')
pickle.dump(lsq_fit_data, f)
f.close()
'''
# un-pickle
f = open(path_lsqfit_pckl, 'rb')
lsq_fit_data = pickle.load(f)
f.close()


# %%
fig, ax = plt.subplots(1, 2)
ax[0].imshow(lsq_fit_data.fitted_parameters_vector[:,:,slice,0])
ax[1].imshow(lsq_fit_data.fitted_parameters_vector[:,:,slice,1])
plt.show()

# %% get rois from gmm
tmp_data = deepcopy(np.reshape(lsq_fit_data.fitted_parameters_vector,[nx*ny*nz,lsq_fit_data.fitted_parameters_vector.shape[-1]])) #[mask>0,:]
wmgm_idx = ((tmp_data[:,0]<2.95e-9) & (tmp_data[:,1]>0.05)) | (tmp_data[:,1]>0.5) # indices of WM/GM only

# get roi mask from k-means / GMM clustering 
print('Estimating tissue ROIs from GMM on LSQ fits...', flush=True)
tmp_data[:,0] = tmp_data[:,0]*1e9
tmp_roi, _ = mask_from_gmm(parameter_vector=tmp_data[wmgm_idx,:], n_clusters=n_tissue)  # mask from GMM
roi_mask = np.zeros([nx*ny*nz])
roi_mask[wmgm_idx] = tmp_roi
roi_mask = np.squeeze(np.reshape(roi_mask, [nx,ny,nz]))

# save nii
hdr_rois = nib.load(path_mask).header.copy()
hdr_rois['dim'][0:4] = np.insert(roi_mask.shape, 0, data.ndim)                  # modify header with correct image dimensions
img = nib.Nifti1Image(roi_mask, None, header=hdr_rois)
img.get_data_dtype()
nib.save(img, path_rois_nii) 

fig, ax = plt.subplots(2, 2)
ax[0,0].imshow(data[:,:,slice,0], cmap='gray'); ax[0,0].set_title('b0')
ax[0,1].imshow((roi_mask[:,:,slice]==1) | (roi_mask[:,:,slice]==2)); ax[0,1].set_title('full mask')
ax[1,0].imshow((roi_mask[:,:,slice]==1)); ax[1,0].set_title('k=1')
ax[1,1].imshow((roi_mask[:,:,slice]==2)); ax[1,1].set_title('k=2')
plt.show()



'''
# %% get roi mask from k-means clustering / gmm
tmp_data = np.reshape(lsq_fit_data.fitted_parameters_vector,[nx*ny*nz,lsq_fit_data.fitted_parameters_vector.shape[-1]]) #[mask>0,:]
tmp_data = tmp_data[np.ndarray.flatten(mask>0),:]
# tmp_data = np.hstack((tmp_data,np.reshape(data,[nx*ny*nz,ndw])[np.ndarray.flatten(mask>0),0][:,np.newaxis]))
# tmp_roi = mask_from_kmeans(parameter_vector=tmp_data, n_clusters=n_tissue+1)
tmp_data[:,0] = tmp_data[:,0]*1e9
tmp_roi, _ = mask_from_gmm(parameter_vector=tmp_data, n_clusters=n_tissue)
# roi_mask = np.reshape(roi_mask,[nx,ny,nz])
roi_mask = np.zeros([nx*ny*nz])
roi_mask[np.ndarray.flatten(mask>0)] = tmp_roi

# roi_mask2 = mask_from_kmeans(parameter_vector=lsq_fit_data2.parameter_vector, n_clusters=n_tissue)

# data = np.squeeze(np.reshape(data, [nx,ny,nz,ndw]))
# mask = np.squeeze(np.reshape(mask, [nx,ny,nz]))
roi_mask = np.squeeze(np.reshape(roi_mask, [nx,ny,nz]))

fig, ax = plt.subplots(2, 4)
ax[0,0].imshow(data[:,:,slice,0], cmap='gray'); ax[0,0].set_title('b0')
ax[0,1].imshow(mask[:,:,slice]); ax[0,1].set_title('full mask')
ax[0,2].imshow(roi_mask[:,:,slice]); ax[0,2].set_title('ROI mask (k=0:4)')
# ax[0,3].imshow(((roi_mask[:,:,slice]==1).astype(int)+(roi_mask[:,:,slice]==3).astype(int))); ax[0,3].set_title('k=2+3')
ax[0,3].imshow(((roi_mask[:,:,slice]==0))); ax[0,3].set_title('k=0')
ax[1,0].imshow((roi_mask[:,:,slice]==1)); ax[1,0].set_title('k=1')
ax[1,1].imshow((roi_mask[:,:,slice]==2)); ax[1,1].set_title('k=2')
ax[1,2].imshow((roi_mask[:,:,slice]==3)); ax[1,2].set_title('k=3')
ax[1,3].imshow((roi_mask[:,:,slice]==4)); ax[1,3].set_title('k=4')
plt.show()
'''




# %% LSQ-predicted signals
'''
E_fit_sm = np.squeeze(sz_sm.simulate_signal(acq_scheme, lsq_fit_data.fitted_parameters_vector))
# pickle
f = open(path_Efit_pckl, 'wb')
pickle.dump(E_fit_sm, f)
f.close()
'''
# un-pickle
f = open(path_Efit_pckl, 'rb')
E_fit_sm = pickle.load(f)
f.close()

plt.plot(E_fit_sm[100,100,70,:])

# %% hierarchical Bayesian fitting
fit_bayes_roi_update_new = reload(fit_bayes_roi_update_new)

nsteps = 2000
burn_in = 1000
nupdates = 20
proc_start = time.time()
# parameters_dict_bayes, acceptance_rate, parameter_convergence, likelihood, weights\
    # = fit_bayes_new.fit(sz_sm, acq_scheme, data_sm, E_fit_sm, lsq_fit_data.fitted_parameters, roi_mask, nsteps, burn_in, nupdates)
parameter_dict = lsq_fit_data.fitted_parameters
for key in lsq_fit_data.fitted_parameters.keys():
    parameter_dict[key] = lsq_fit_data.fitted_parameters[key][:,:,slice]
    print(parameter_dict[key].shape)
model=sz_sm
acq_scheme=acq_scheme
data=data_sm[:,:,slice,:]
E_fit=E_fit_sm[:,:,slice,:]
parameter_vector_init=parameter_dict
mask=roi_mask[:,:,slice]
nsteps=nsteps
burn_in=burn_in
nupdate=nupdates

parameters_dict_bayes, acceptance_rate, parameter_convergence, likelihood, weights \
    = fit_bayes_roi_update.fit(model=sz_sm, acq_scheme=acq_scheme, data=data_sm[:,:,slice,:],\
                               E_fit=E_fit_sm[:,:,slice,:], parameter_vector_init=parameter_dict,\
                               mask=roi_mask[:,:,slice], nsteps=nsteps, burn_in=burn_in, nupdate=nupdate)

compute_time(proc_start, time.time())


# %%
parameters_vect_bayes = np.zeros((nx*ny,parameters_dict_bayes.keys().__len__()))
i=0
for key in parameters_dict_bayes.keys():
    parameters_vect_bayes[:,i] = parameters_dict_bayes.get(key)
    i+=1
parameters_vect_bayes = np.reshape(parameters_vect_bayes,[nx,ny,parameters_dict_bayes.keys().__len__()])

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(lsq_fit_data.fitted_parameters_vector[:,:,slice,0]); ax[0, 0].set_title('LSQ, ')
ax[0, 1].imshow(lsq_fit_data.fitted_parameters_vector[:,:,slice,1]); ax[0, 1].set_title('LSQ, ')
ax[1, 0].imshow(parameters_vect_bayes[:,:,0]); ax[1, 0].set_title('Bayes, ')
ax[1, 1].imshow(parameters_vect_bayes[:,:,1]); ax[1, 1].set_title('Bayes, ')
plt.show()

fig, ax = plt.subplots(1, 3)
voxels = np.random.randint(low=0, high=acceptance_rate.shape[0], size=20, dtype=int)
for vox in voxels:
    ax[0].plot(acceptance_rate[vox,:]); ax[0].set_title('acceptance rate')
    ax[1].plot(parameter_convergence['BundleModel_1_G2Zeppelin_1_lambda_par'][vox,:]); ax[1].set_title('diffusivity')
    ax[2].plot(parameter_convergence['BundleModel_1_partial_volume_0'][vox,:]); ax[2].set_title('vol frac')


