#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:03:52 2024

@author: epowell
"""

basedir = os.path.join('/home','epowell','code','python','dmipy-bayesian')      # path to matlab-bayesian code directory
loadloc = basedir #os.path.join(basedir, 'data')
saveloc = basedir #os.path.join(basedir, 'data')

# save simulated data for example
data_shelf = shelve.open(os.path.join(loadloc,'sim_dki_data'))                  # vars: adc, bvals, kur, mask, nvox, nx, ny, rois, s0, t2, te
for key in ['adc','bvals','bvecs','kur','mask','nvox','nx','ny','rois','s0','t2','te']:
    try:
        print(key)
        data_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
data_shelf.close()


# load simulated data for example
data_shelf = shelve.open(os.path.join(loadloc,'sim_dki_data'))                  # vars: adc, bvals, kur, mask, nvox, nx, ny, rois, s0, t2, te
for key in data_shelf:
    globals()[key]=data_shelf[key]
data_shelf.close()


# save output
data_shelf = shelve.open(os.path.join(saveloc,'result_nsteps1e5'),'n')          # note: loading model and lsq_fit depends on finding gaussian_models.py / kurtosis_model.py in current directory
for key in ['params_all','acceptance_rate','param_conv','acq_scheme','E_noisy',
            'E_nfree','parameter_dict_init','rois','nsteps','burn_in','nupdates',
            'SNR','red']:
    try:
        print(key)
        data_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
data_shelf.close()
        