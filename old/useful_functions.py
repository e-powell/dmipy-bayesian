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
import sklearn
from sklearn.cluster import KMeans

from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti


# to temporarily suppress output to console
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def make_square_axes(ax):
    '''
    Make axes square
    '''
    ax.set_aspect(1 / ax.get_data_ratio())


def compute_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("/nTOTAL OPTIMIZATION TIME {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    time_string = ("TOTAL TIME {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    return time_string


def compute_temp_schedule(startTemp, endTemp, MAX_ITER):
    SA_schedule = []
    it = np.arange(MAX_ITER + 1)
    SA_schedule.append([-math.log(endTemp / startTemp) / i for i in it])
    return SA_schedule[0]


def save_workspace(filename):
    '''
    Save workspace variables
    '''
    print(filename)
    shelf = shelve.open(filename, "n")
    for key in globals():
        try:
            # print(key)
            shelf[key] = globals()[key]
        except:
            print('ERROR shelving: {0}'.format(key))
    shelf.close()


def load_workspace(filename):
    '''
    Load workspace variables
    '''
    shelf = shelve.open(filename)
    print(shelf)
    for key in shelf:
        try:
            print(key)
            globals()[key] = shelf[key]
        except:
            print('ERROR shelving: {0}'.format(key))
    shelf.close()


def cart2mu(xyz):
    '''
    Cartesian coords to polar angles
    '''
    shape = xyz.shape[:-1]
    mu = np.zeros(np.r_[shape, 2])
    r = np.linalg.norm(xyz, axis=-1)
    mu[..., 0] = np.arccos(xyz[..., 2] / r)  # theta
    mu[..., 1] = np.arctan2(xyz[..., 1], xyz[..., 0])
    mu[r == 0] = 0, 0
    return mu


def create_spherical_mean_scheme(acq_scheme):
    '''
    Create spherical mean scheme from directional scheme
    
    Parameters
    ----------
    acq_scheme : dmipy acquisition scheme
    '''
    acq_scheme_smt = acq_scheme.spherical_mean_scheme
    # create fake gradient directions
    grad_dirs = np.tile([1,1,1] / np.linalg.norm([1,1,1]), [np.unique(acq_scheme.bvalues).shape[0], 1])
    # grad_dirs = np.tile([0,0,0], [np.unique(acq_scheme.bvalues).shape[0], 1])
    acq_scheme_smt = acquisition_scheme_from_bvalues(np.unique(acq_scheme.bvalues), grad_dirs, acq_scheme.delta[0], acq_scheme.Delta[0])
    # acq_scheme_smt.gradient_directions = grad_dirs
    
    return acq_scheme_smt


def add_noise(data, snr=50):
    '''
    Add noise to signals
    
    Parameters
    ----------
    data : float,
        array of signals [any dimension]
    snr: float, 
        SNR of data; assumes np.max(data) == 1
    '''
    data_real = data + np.random.normal(scale=1/snr, size=np.shape(data))
    data_imag = np.random.normal(scale=1/snr, size=np.shape(data))
    data_noisy = np.sqrt(data_real**2 + data_imag**2)

    return data_noisy


def mask_from_tensor_model(signals, acq_scheme):
    '''
    Create ROI mask using tensor model fit to signals
    
    Parameters
    ----------
    signals : float,
        array of signals [nvox, ndwi]
    acq_scheme : dmipy acquisition scheme
    '''
    
    # set up the dipy aquisition
    gtab = gradient_table(acq_scheme.bvalues, acq_scheme.gradient_directions)
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(signals)
    
    # threshold md and fa to estimate the ROI mask
    md_thresh = 1.5e-9
    fa_thresh = 0.8

    roi_mask = np.zeros(signals.shape[0])

    # white matter - less than md threshold and higher than fa threshold
    roi_mask[(tenfit.md < md_thresh) & (tenfit.fa > fa_thresh)] = 1
    # grey matter - less than md threshold and less than fa threshold
    roi_mask[(tenfit.md < md_thresh) & (tenfit.fa < fa_thresh)] = 2
    # csf - higher than md threshold and lower than fa threshold
    roi_mask[(tenfit.md > md_thresh) & (tenfit.fa < fa_thresh)] = 3
    
    # plt.plot(tenfit.fa)
    # plt.plot(tenfit.md)
    
    return roi_mask


def mask_from_kmeans(parameter_vector, n_clusters):
    '''
    Create ROI mask using k-means clustering of LSQ parameters
    
    Parameters
    ----------
    parameter_vector : float,
        array of LSQ-derived parameters [nvox, nparams]
    n_clusters : int, 
        number of clusters (i.e. tissue types) to find
    '''

    # cluster the voxels into ROIs using kmeans - could do this iteratively?
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(parameter_vector)

    roi_mask = kmeans.labels_ + 1
    
    return roi_mask

def mask_from_gmm(parameter_vector, n_clusters):
    '''
    Create ROI mask using Gaussian mixture model
    
    Parameters
    ----------
    parameter_vector : float,
        array of LSQ-derived parameters [nvox, nparams]
    n_clusters : int, 
        number of clusters (i.e. tissue types) to find
    '''

    # training gaussian mixture model
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(parameter_vector)
    
    # predictions from gmm
    roi_mask = gmm.predict(parameter_vector) + 1
    
    return roi_mask, gmm


def check_lsq_fit(model, parameters_lsq_dict, mask=None):
    '''
    Check no LSQ fits hit boundaries; add/sub eps if so
    
    Parameters
    ----------
    model : dmipy model
    parameters_lsq_dict : dictionary,
        fitted parameters, e.g. from dmipy_model.fitted_parameters
    '''
    
    # set mask default
    if mask is None:
        mask = np.ones(data[:, 0] > 0).astype('uint8')
    # convert to int if input as bool
    elif mask.any():
        mask = mask.astype('uint8')
        
    nvox = 0
    for param in model.parameter_names:  
        # set up default mask
        if mask is None:
            mask = np.ones(np.prod(parameters_lsq_dict[param].shape), dtype=bool)
        else:
            mask = np.reshape(mask.astype(dtype=bool), np.prod(parameters_lsq_dict[param].shape))
            
        if model.parameter_cardinality[param] > 1:
            for card in range(model.parameter_cardinality[param]):
                idx = parameters_lsq_dict[param][:, card] <= model.parameter_ranges[param][card][0] * model.parameter_scales[param]
                nvox += idx[mask].sum()
                parameters_lsq_dict[param][idx, card] = (model.parameter_ranges[param][card][0] + np.finfo(float).eps) * model.parameter_scales[param][card]
                idx = parameters_lsq_dict[param][:, card] >= model.parameter_ranges[param][card][1] * model.parameter_scales[param]
                nvox += idx[mask].sum()
                parameters_lsq_dict[param][idx, card] = (model.parameter_ranges[param][card][1] - np.finfo(float).eps) * model.parameter_scales[param][card]
        elif model.parameter_cardinality[param] == 1:
            idx = parameters_lsq_dict[param] <= model.parameter_ranges[param][0] * model.parameter_scales[param]
            nvox += idx[mask].sum()
            parameters_lsq_dict[param][idx] = (model.parameter_ranges[param][0] + np.finfo(float).eps) * model.parameter_scales[param]
            idx = parameters_lsq_dict[param] >= model.parameter_ranges[param][1] * model.parameter_scales[param]
            nvox += idx[mask].sum()
            parameters_lsq_dict[param][idx] = (model.parameter_ranges[param][1] - np.finfo(float).eps) * model.parameter_scales[param]
   
    return parameters_lsq_dict, nvox

