# -*- coding: utf-8 -*-
'''
Document Module
'''
from __future__ import division

import numpy as np
from scipy.special import erf
from scipy.stats import norm  # ecap

from ..utils import utils
from ..core.modeling_framework import ModelProperties
from ..core.signal_model_properties import (
    IsotropicSignalModelProperties, AnisotropicSignalModelProperties)
from dipy.utils.optpkg import optional_package

numba, have_numba, _ = optional_package("numba")

DIFFUSIVITY_SCALING = 1e-9
A_SCALING = 1e-12

__all__ = [
    'G1Ball',
    'G1BallNormalDist',
    'G2BallNormalDist',
    'G2Zeppelin',
    'G3TemporalZeppelin'
]


class G1Ball(ModelProperties, IsotropicSignalModelProperties):
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
        'lambda_iso': (.1, 3)
    }
    _parameter_scales = {
        'lambda_iso': DIFFUSIVITY_SCALING
    }
    _parameter_types = {
        'lambda_iso': 'normal',
    }
    _model_type = 'CompartmentModel'

    def __init__(self, lambda_iso=None):
        self.lambda_iso = lambda_iso

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
        E_ball = np.exp(-bvals * lambda_iso)
        return E_ball


# paddy start
class G1BallNormalDist(ModelProperties, IsotropicSignalModelProperties):
    r""" Like the Ball model [1]_ - an isotropic Tensor with one diffusivity,
    but with a normal distribution of diffusivities (rather than a single value)
    parameterised by mean mu and standard deviation sigma. I.e. the diffusion
    equivalent of [2].



    Parameters
    ----------
    lambda_iso_mean : float,
        mean of the isotropic diffusivity in m^2/s.

    lambda_iso_std : float,
        standard deviation of the isotropic diffusivity in m^2/s.

    References
    ----------
    .. [1] Behrens et al.
           "Characterization and propagation of uncertainty in
            diffusion-weighted MR imaging"
           Magnetic Resonance in Medicine (2003)
    .. [2] Devine et al.
           "Simplified Luminal Water Imaging for the
           Detection of Prostate Cancer From Multiecho
           T2 MR Images,
           Journal of Magnetic Resonance Imaging (2018)
    """

    _required_acquisition_parameters = ['bvalues']

    _parameter_ranges = {
        'lambda_iso_mean': (.1, 3),
        'lambda_iso_std': (.01, 0.5)
    }
    _parameter_scales = {
        'lambda_iso_mean': DIFFUSIVITY_SCALING,
        'lambda_iso_std': DIFFUSIVITY_SCALING
    }
    _parameter_types = {
        'lambda_iso_mean': 'normal',
        'lambda_iso_std': 'normal',
    }
    _model_type = 'CompartmentModel'

    def __init__(self, lambda_iso_mean=None, lambda_iso_std=None):
        self.lambda_iso_mean = lambda_iso_mean
        self.lambda_iso_std = lambda_iso_std

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
        lambda_iso_mean = kwargs.get('lambda_iso_mean', self.lambda_iso_mean)
        lambda_iso_std = kwargs.get('lambda_iso_std', self.lambda_iso_std)

        r'''
        define the grid (on which to estimate the distribution) based on
        the minimum and maximum lambda_iso values
        '''
        # lambda_iso_grid = np.linspace(.1 * DIFFUSIVITY_SCALING, 3 * DIFFUSIVITY_SCALING, 1000)
        lambda_iso_grid = np.linspace(self.parameter_ranges['lambda_iso_mean'][0],
                                      self.parameter_ranges['lambda_iso_mean'][1], 1000)  # * self.parameter_scales['lambda_iso_mean']

        r'''
        make the distribution and normalise it
        '''
        normaldist = norm.pdf(lambda_iso_grid, lambda_iso_mean, lambda_iso_std)
        # normaldist = (1/(lambda_iso_std*np.sqrt(2*np.pi))) * np.exp(-.5*((lambda_iso_grid-lambda_iso_mean)/lambda_iso_std)**2)
        normaldist = normaldist / np.sum(normaldist)
        # ecap help: hard code norm pdf

        r'''calculate the signal'''
        E_ball = np.matmul(self.signal_dictionary, normaldist)

        return E_ball


# paddy stop


# ecap start
class G2BallNormalDist(ModelProperties, IsotropicSignalModelProperties):
    r""" Like the G1BallNormalDist model [1]_ - an isotropic Tensor with one diffusivity,
    but with a normal distribution of diffusivities (rather than a single value)
    parameterised by mean mu and standard deviation sigma. I.e. the diffusion
    equivalent of [2] - but with 2 compartments

    also: diffusivity == T2
          b-value     == TE

    Parameters
    ----------
    lambda_iso_mean : float, [2x1]
        mean of the isotropic diffusivity in m^2/s, of compartments 1 & 2

    lambda_iso_std : float, [2x1]
        standard deviation of the isotropic diffusivity in m^2/s, of compartments 1 & 2

    dist_height : float,
        relative height of the distribution (compartment 1)

    References
    ----------
    .. [1] Behrens et al.
           "Characterization and propagation of uncertainty in
            diffusion-weighted MR imaging"
           Magnetic Resonance in Medicine (2003)
    .. [2] Devine et al.
           "Simplified Luminal Water Imaging for the
           Detection of Prostate Cancer From Multiecho
           T2 MR Images,
           Journal of Magnetic Resonance Imaging (2018)
    """

    _required_acquisition_parameters = ['bvalues']

    _parameter_ranges = {
        'lambda_iso_mean_0': (.1, 3),
        'lambda_iso_mean_1': (.1, 3),
        'lambda_iso_std_0': (.01, 0.5),
        'lambda_iso_std_1': (.01, 0.5),
        'dist_height': (0, 1)
    }
    _parameter_scales = {
        'lambda_iso_mean_0': DIFFUSIVITY_SCALING,
        'lambda_iso_mean_1': DIFFUSIVITY_SCALING,
        'lambda_iso_std_0': DIFFUSIVITY_SCALING,
        'lambda_iso_std_1': DIFFUSIVITY_SCALING,
        'dist_height': 1
    }
    _parameter_types = {
        'lambda_iso_mean_0': 'normal',
        'lambda_iso_mean_1': 'normal',
        'lambda_iso_std_0': 'normal',
        'lambda_iso_std_1': 'normal',
        'dist_height': 'normal',
    }
    _model_type = 'CompartmentModel'

    def __init__(self, lambda_iso_mean_0=None, lambda_iso_std_0=None, lambda_iso_mean_1=None, lambda_iso_std_1=None,
                 dist_height=None):
        self.lambda_iso_mean_0 = lambda_iso_mean_0
        self.lambda_iso_std_0 = lambda_iso_std_0
        self.lambda_iso_mean_1 = lambda_iso_mean_1
        self.lambda_iso_std_1 = lambda_iso_std_1
        self.dist_height = dist_height

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
        lambda_iso_mean_0 = kwargs.get('lambda_iso_mean_0', self.lambda_iso_mean_0)
        lambda_iso_std_0 = kwargs.get('lambda_iso_std_0', self.lambda_iso_std_0)
        lambda_iso_mean_1 = kwargs.get('lambda_iso_mean_1', self.lambda_iso_mean_1)
        lambda_iso_std_1 = kwargs.get('lambda_iso_std_1', self.lambda_iso_std_1)
        dist_height = kwargs.get('dist_height', self.dist_height)

        r'''
        define the grid (on which to estimate the distribution) based on
        the minimum and maximum lambda_iso values
        '''
        lambda_iso_grid_0 = np.linspace(self.parameter_ranges['lambda_iso_mean_0'][0],
                                        self.parameter_ranges['lambda_iso_mean_0'][1], 1000) \
                            * self.parameter_scales['lambda_iso_mean_0']
        lambda_iso_grid_1 = np.linspace(self.parameter_ranges['lambda_iso_mean_1'][0],
                                        self.parameter_ranges['lambda_iso_mean_1'][1], 1000) \
                            * self.parameter_scales['lambda_iso_mean_1']

        r'''
        make the distribution and normalise it
        '''
        # print(lambda_iso_grid_1)
        # print(lambda_iso_mean_0)
        # print(lambda_iso_std_0)
        normaldist_0 = norm.pdf(lambda_iso_grid_0, lambda_iso_mean_0, lambda_iso_std_0)
        normaldist_1 = norm.pdf(lambda_iso_grid_1, lambda_iso_mean_1, lambda_iso_std_1)

        r'''adjust by the distribution height'''
        normaldist = (dist_height * normaldist_0) + ((1 - dist_height) * normaldist_1)

        r'''calculate the signal'''
        # E_ball_1 = np.matmul(self.signal_dictionary, normaldist_1)
        # E_ball_2 = np.matmul(self.signal_dictionary, normaldist_2)
        E_ball = np.matmul(self.signal_dictionary, normaldist)

        r'''adjust by the distribution height'''
        # E_ball = (dist_height * E_ball_1) + ((1 - dist_height) * E_ball_2)

        return E_ball


# ecap stop

class G2Zeppelin(ModelProperties, AnisotropicSignalModelProperties):
    r""" The Zeppelin model [1]_ - an axially symmetric Tensor - typically used
    for extra-axonal diffusion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in m^2/s.
    lambda_perp : float,
        perpendicular diffusivity in m^2/s.

    Returns
    -------
    E_zeppelin : float or array, shape(N),
        signal attenuation.

    References
    ----------
    .. [1] Panagiotaki et al.
           "Compartment models of the diffusion MR signal in brain white
            matter: a taxonomy and comparison". NeuroImage (2012)
    """
    _required_acquisition_parameters = ['bvalues', 'gradient_directions']

    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'lambda_perp': (.1, 3)
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
        'lambda_perp': DIFFUSIVITY_SCALING
    }
    _parameter_types = {
        'mu': 'orientation',
        'lambda_par': 'normal',
        'lambda_perp': 'normal',
    }
    _model_type = 'CompartmentModel'

    def __init__(self, mu=None, lambda_par=None, lambda_perp=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.lambda_perp = lambda_perp

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
        n = acquisition_scheme.gradient_directions
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)
        mu = kwargs.get('mu', self.mu)
        mu = utils.unitsphere2cart_1d(mu)
        E_zeppelin = _attenuation_zeppelin(
            bvals, lambda_par, lambda_perp, n, mu)
        return E_zeppelin

    def spherical_mean(self, acquisition_scheme, **kwargs):
        """
        Estimates spherical mean for every shell in acquisition scheme for
        Zeppelin model.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        E_mean : float,
            spherical mean of the Zeppelin model for every acquisition shell.
        """
        bvals = acquisition_scheme.shell_bvalues[
            ~acquisition_scheme.shell_b0_mask]

        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)

        E_mean = np.ones_like(acquisition_scheme.shell_bvalues)
        if lambda_par > lambda_perp:  # use [kaden et al. 2016]
            exp_bl = np.exp(-bvals * lambda_perp)
            sqrt_bl = np.sqrt(bvals * (lambda_par - lambda_perp))
            E_mean_ = exp_bl * np.sqrt(np.pi) * erf(sqrt_bl) / (2 * sqrt_bl)
            E_mean[~acquisition_scheme.shell_b0_mask] = E_mean_
        else:  # estimate spherical mean using rotational harmonics
            rh_array = self.rotational_harmonics_representation(
                acquisition_scheme, **kwargs)
            E_mean[acquisition_scheme.unique_dwi_indices] = (
                    rh_array[:, 0] / (2 * np.sqrt(np.pi))
            )
        return E_mean


class G3TemporalZeppelin(ModelProperties, AnisotropicSignalModelProperties):
    r"""
    The temporal Zeppelin model [1]_ - an axially symmetric Tensor - typically
    used to describe extra-axonal diffusion. The G3TemporalZeppelin differs
    from G2Zeppelin in that it has a time-dependent perpendicular parameter
    "A", which describe extra-axonal diffusion hindrance due to axon packing,
    and that lambda_perp is instead called lambda_inf, as it describes the
    perpendicular diffusivity when diffusion time is infinite.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.
    lambda_inf : float,
        bulk diffusivity constant 10^9 m^2/s.
    A: float,
        characteristic coefficient in 10^12 m^2

    Returns
    -------
    E_zeppelin : float or array, shape(N),
        signal attenuation.

    References
    ----------
    .. [1] Burcaw, L.M., Fieremans, E., Novikov, D.S., 2015. Mesoscopic
        structure of neuronal tracts from time-dependent diffusion.
        NeuroImage 114, 18.
    """
    _required_acquisition_parameters = [
        'bvalues', 'gradient_directions', 'delta', 'Delta']

    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'lambda_inf': (.1, 3),
        'A': (0, 10)
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
        'lambda_inf': DIFFUSIVITY_SCALING,
        'A': A_SCALING
    }
    _parameter_types = {
        'mu': 'orientation',
        'lambda_par': 'normal',
        'lambda_inf': 'normal',
        'A': 'normal'
    }
    _model_type = 'CompartmentModel'

    def __init__(self, mu=None, lambda_par=None, lambda_inf=None, A=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.lambda_inf = lambda_inf
        self.A = A

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
        n = acquisition_scheme.gradient_directions
        delta = acquisition_scheme.delta
        Delta = acquisition_scheme.Delta

        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_inf = kwargs.get('lambda_inf', self.lambda_inf)
        A = kwargs.get('A', self.A)
        mu = kwargs.get('mu', self.mu)
        mu = utils.unitsphere2cart_1d(mu)

        R1 = mu
        R2 = utils.perpendicular_vector(R1)
        R3 = np.cross(R1, R2)
        R = np.c_[R1, R2, R3]

        E_zeppelin = np.ones_like(bvals)
        for i, (bval_, n_, delta_, Delta_) in enumerate(
                zip(bvals, n, delta, Delta)
        ):
            # lambda_perp and A must be in the same unit
            restricted_term = (
                    A * (np.log(Delta_ / delta_) + 3 / 2.) / (Delta_ - delta_ / 3.)
            )
            D_perp = lambda_inf + restricted_term
            D_h = np.diag(np.r_[lambda_par, D_perp, D_perp])
            D = np.dot(np.dot(R, D_h), R.T)
            E_zeppelin[i] = np.exp(-bval_ * np.dot(n_, np.dot(n_, D)))
        return E_zeppelin

    def spherical_mean(self, acquisition_scheme, **kwargs):
        """
        Estimates spherical mean for every shell in acquisition scheme for
        Restricted Zeppelin model.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        E_mean : float,
            spherical mean of the Restricted Zeppelin model for every
            acquisition shell.
        """
        bvals = acquisition_scheme.shell_bvalues[
            ~acquisition_scheme.shell_b0_mask]
        delta = acquisition_scheme.shell_delta[
            ~acquisition_scheme.shell_b0_mask]
        Delta = acquisition_scheme.shell_Delta[
            ~acquisition_scheme.shell_b0_mask]
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_inf = kwargs.get('lambda_inf', self.lambda_inf)
        A = kwargs.get('A', self.A)
        E_mean = np.ones_like(acquisition_scheme.shell_bvalues)

        restricted_term = (
                A * (np.log(Delta / delta) + 3 / 2.) / (Delta - delta / 3.)
        )
        lambda_perp = lambda_inf + restricted_term
        if lambda_par > lambda_perp.max():  # use modified [kaden et al. 2016]
            exp_bl = np.exp(-bvals * lambda_perp)
            sqrt_bl = np.sqrt(bvals * (lambda_par - lambda_perp))
            E_mean[~acquisition_scheme.shell_b0_mask] = (
                    exp_bl * np.sqrt(np.pi) * erf(sqrt_bl) / (2 * sqrt_bl))
        else:  # estimate spherical mean using rotational harmonics
            rh_array = self.rotational_harmonics_representation(
                acquisition_scheme, **kwargs)
            E_mean[acquisition_scheme.unique_dwi_indices] = (
                    rh_array[:, 0] / (2 * np.sqrt(np.pi))
            )
        return E_mean


def _attenuation_zeppelin(bvals, lambda_par, lambda_perp, n, mu):
    "Signal attenuation for Zeppelin model."
    mu_perpendicular_plane = np.eye(3) - np.outer(mu, mu)
    magnitude_parallel = np.dot(n, mu)
    proj = np.dot(mu_perpendicular_plane, n.T)
    magnitude_perpendicular = np.sqrt(
        proj[0] ** 2 + proj[1] ** 2 + proj[2] ** 2)
    E_zeppelin = np.exp(-bvals *
                        (lambda_par * magnitude_parallel ** 2 +
                         lambda_perp * magnitude_perpendicular ** 2)
                        )
    return E_zeppelin


if have_numba:
    _attenuation_zeppelin = numba.njit()(_attenuation_zeppelin)
