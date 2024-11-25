# -*- coding: utf-8 -*-
'''
Author: Paddy Slator
'''
from __future__ import division

import numpy as np
from scipy.special import erf

from dmipy import utils
from dmipy.core.modeling_framework import ModelProperties
from dmipy.core.signal_model_properties import (
    IsotropicSignalModelProperties, AnisotropicSignalModelProperties)
from dipy.utils.optpkg import optional_package

numba, have_numba, _ = optional_package("numba")

DIFFUSIVITY_SCALING = 1e-9
A_SCALING = 1e-12

__all__ = [
    'G1Kurtosis',
]


class G1Kurtosis(ModelProperties, IsotropicSignalModelProperties):
    r""" The Ball model [1]_ - an isotropic Tensor with one diffusivity.

    Parameters
    ----------
    lambda_iso : float,
        isotropic diffusivity in m^2/s.

    References
    ----------
    .. [1] Behrens et al.
           "Characterization and propagation of uncertainty in
            diffusion-weighted MR imaging"
           Magnetic Resonance in Medicine (2003)
    """
    _required_acquisition_parameters = ['bvalues']

    _parameter_ranges = {
        'lambda_iso': (.1, 3),
        'kurtosis': (0.001,2)
    }
    _parameter_scales = {
        'lambda_iso': DIFFUSIVITY_SCALING,
        'kurtosis': 1.0
    }
    _parameter_types = {
        'lambda_iso': 'normal',
        'kurtosis': 'normal'
    }
    _model_type = 'CompartmentModel'

    def __init__(self, lambda_iso=None, kurtosis=None):
        self.lambda_iso = lambda_iso
        self.kurtosis = kurtosis

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Estimates the signal attenuation.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''
        bvals = acquisition_scheme.bvalues
        lambda_iso = kwargs.get('lambda_iso', self.lambda_iso)
        kurtosis = kwargs.get('kurtosis', self.kurtosis)

        E_kurtosis = np.exp(-bvals * lambda_iso + (1/6) * bvals**2 * lambda_iso**2 * kurtosis)



        return E_kurtosis
