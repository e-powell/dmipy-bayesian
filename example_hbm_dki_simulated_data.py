#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:16:48 2024

@author: epowell
"""
import matplotlib.pyplot as plt
import numpy as np
import os, pdb, shelve
#import os
#import shelve
import fit_bayes, kurtosis_model
from copy import copy, deepcopy
from datetime import datetime
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.data import saved_acquisition_schemes
#import fit_bayes
from importlib import reload
#import pdb
from matplotlib.colors import Normalize

#-------------------------- USER DEFINED PARAMETERS --------------------------#
basedir = os.path.join('/home','epowell','code','python','dmipy-bayesian')      # path to matlab-bayesian code directory
nsteps = int(1e3)                                                               # no. MCMC steps
burn_in = nsteps // 2                                                           # no. of steps to discard (burn-in)
nupdates = int(100)                                                             # how often weights are updated
nroifit = 2                                                                     # no. regional priors
SNR = 20                                                                        # SNR in b=0 data
nsa = 1                                                                         # no. signal averages
ninit = 25                                                                      # no. initial values for LSQ fit
red = .1                                                                        # proportion of original HCP protocol to retain
saveflag = 1                                                                    # save output (1 yes; 0 no)
#-----------------------------------------------------------------------------#

# fitting bounds
lb = [.1e-9, 0, 1]                                                              # D, K, s0
ub = [3.5e-9, 3, 1]                                                             # D, K, s0

# load test data
loadloc = basedir
data_shelf = shelve.open(os.path.join(loadloc,'sim_dki_data'))                  # vars: adc, bvals, kur, mask, nvox, nx, ny, rois, s0, t2, te
for key in data_shelf:
    globals()[key]=data_shelf[key]
data_shelf.close()

unib = np.unique(bvals)

# reduce protocol size
if red is not None and red != 1:
    bvals = bvals[:int(np.round(len(bvals) * red))]
    bvecs = bvecs[:int(np.round(len(bvecs) * red))]

if saveflag:
    saveloc = loadloc
    savenm = 'sim_dki_fit__' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print('Saving to ' + os.path.join(saveloc,savenm))


# %% Create model, parameter dictionary, and simulate signals (noise-free and noisy)

# acquisition scheme and model (dmipy-style)
acq_scheme = acquisition_scheme_from_bvalues(bvalues=bvals,gradient_directions=bvecs)
model = MultiCompartmentModel([kurtosis_model.G1Kurtosis()])
model.parameter_fixed = dict.fromkeys(model.parameter_names)
for param in model.parameter_names:
    model.parameter_fixed[param] = 0
# create parameter dictionary (ground truth)
param_dict = dict.fromkeys(model.parameter_names)
param_dict['G1Kurtosis_1_lambda_iso'] = adc
param_dict['G1Kurtosis_1_kurtosis'] = kur
# simulate signals (noise-free and noisy)
E_nfree = model.simulate_signal(acquisition_scheme=acq_scheme,parameters_array_or_dict=param_dict)
E_noisy = np.zeros((nvox, len(bvals)))
for i in range(nvox):
    E_noisy[i, :] = (E_nfree[i, :] / np.mean(E_nfree[i, bvals == 0])) + (1 / (SNR * np.sqrt(nsa))) * np.random.randn(len(bvals))


# %% LSQ fitting
lsq_fit = model.fit(acq_scheme, E_noisy, mask)


# %% HBM fitting
#pdb.set_trace()
fit_bayes = reload(fit_bayes)
parameter_dict_init = deepcopy(lsq_fit.fitted_parameters)
parameter_dict_init['G1Kurtosis_1_lambda_iso'] *= 1e9
print(parameter_dict_init['G1Kurtosis_1_lambda_iso'])
params_all, acceptance_rate, param_conv = fit_bayes.fit(model, acq_scheme, E_noisy, parameter_dict_init, None, np.array(rois), nsteps, burn_in, nupdates, 0)


# %% saving

if saveflag:
    data_shelf = shelve.open(os.path.join(saveloc,savenm),'n')
    for key in ['params_all','acceptance_rate','param_conv','acq_scheme','E_noisy',
                'E_nfree','parameter_dict_init','rois','nsteps','burn_in','nupdates',
                'SNR','red']:
        try:
            print(key)
            data_shelf[key] = globals()[key]
        except TypeError:                                                       # __builtins__, my_shelf, and imported modules can not be shelved.
            print('ERROR shelving: {0}'.format(key))
    data_shelf.close()

# %% plotting

nr, nc = 3, 2

col_adc = plt.cm.autumn
col_kur = plt.cm.viridis  # Substitute for parula colormap

lb_plot = lb
ub_plot = ub

fig, axes = plt.subplots(nr, nc, figsize=(15, 15))

# Plot GT
tmp = np.reshape(np.array(adc) * 1e9, (nx, ny))
ax = axes[0, 0]
im = ax.imshow(tmp, norm=Normalize(vmin=lb_plot[0] * 1e9, vmax=ub_plot[0] * 1e9), cmap=col_adc)
fig.colorbar(im, ax=ax, label='[um^2/mm]')
ax.set_title('D (GT)')
ax.axis('off')

tmp = np.reshape(kur, (nx, ny))
ax = axes[0, 1]
im = ax.imshow(tmp, norm=Normalize(vmin=lb_plot[1], vmax=ub_plot[1]), cmap=col_kur)
fig.colorbar(im, ax=ax, label='[a.u.]')
ax.set_title('K (GT)')
ax.axis('off')

# Plot LSQ fit
tmp = np.reshape(lsq_fit.fitted_parameters['G1Kurtosis_1_lambda_iso'] * 1e9, (nx, ny))
ax = axes[1, 0]
im = ax.imshow(tmp, norm=Normalize(vmin=lb_plot[0] * 1e9, vmax=ub_plot[0] * 1e9), cmap=col_adc)
fig.colorbar(im, ax=ax, label='[um^2/mm]')
ax.set_title('D (LSQ)')
ax.axis('off')

tmp = np.reshape(lsq_fit.fitted_parameters['G1Kurtosis_1_kurtosis'], (nx, ny))
ax = axes[1, 1]
im = ax.imshow(tmp, norm=Normalize(vmin=lb_plot[1], vmax=ub_plot[1]), cmap=col_kur)
fig.colorbar(im, ax=ax, label='[a.u.]')
ax.set_title('K (LSQ)')
ax.axis('off')

# Plot Bayesian fit
tmp = np.reshape(params_all['G1Kurtosis_1_lambda_iso'] * 1e9, (nx, ny))
ax = axes[2, 0]
im = ax.imshow(tmp, norm=Normalize(vmin=lb_plot[0] * 1e9, vmax=ub_plot[0] * 1e9), cmap=col_adc)
fig.colorbar(im, ax=ax, label='[um^2/mm]')
ax.set_title('D (HBM)')
ax.axis('off')

tmp = np.reshape(params_all['G1Kurtosis_1_kurtosis'], (nx, ny))
ax = axes[2, 1]
im = ax.imshow(tmp, norm=Normalize(vmin=lb_plot[1], vmax=ub_plot[1]), cmap=col_kur)
fig.colorbar(im, ax=ax, label='[a.u.]')
ax.set_title('K (HBM)')
ax.axis('off')

plt.tight_layout()
plt.show()
