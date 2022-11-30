# load some necessary modules
# from dmipy.core import modeling_framework
# from os.path import join
import scipy.stats
# from os.path import join as pjoin
import numpy as np
from os import listdir
from os.path import join
import nibabel as nib
from copy import copy, deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import transforms
from mpl_toolkits.axes_grid1 import AxesGrid
import time
import math
from importlib import reload
import shelve
from phantominator import shepp_logan, dynamic

# load HCP acqusition scheme
from dmipy.data import saved_acquisition_schemes

# ball stick and spherical mean ball-stick model
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core.modeling_framework import MultiCompartmentSphericalMeanModel, MultiCompartmentModel
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.distributions.distribute_models import SD1WatsonDistributed

import fit_bayes
fit_bayes = reload(fit_bayes)
from fit_bayes import fit, tform_params  # , dict_to_array, array_to_dict


# Make axes square
def make_square_axes(ax):
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


def setup_ballstick():
    acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    nmeas = len(acq_scheme.bvalues)

    stick = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()
    ballstick = MultiCompartmentModel(models=[stick, ball])
    return acq_scheme, nmeas, stick, ball, ballstick


def setup_noddi(lambda_par, lambda_iso):
    acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    nmeas = len(acq_scheme.bvalues)

    stick = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()
    zeppelin = gaussian_models.G2Zeppelin()

    watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
    # set tortuous parameters
    watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par',
                                                   'partial_volume_0')
    watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    # put model together (diffusivities not fixed)
    noddi_full = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])

    # fix diffusivities (stick & zeppelin; ball)
    watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', lambda_par)  # fix diffusivity stick & zeppelin
    noddi = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
    noddi.set_fixed_parameter('G1Ball_1_lambda_iso', lambda_iso)  # fix diffusivity (ball)

    return acq_scheme, nmeas, noddi, noddi_full


def load_real_data(path_ROIs, path_diff, model_to_fit):
    if model_to_fit == 'ballstick':
        acq_scheme, nmeas, stick, ball, model = setup_ballstick()
    elif model_to_fit == 'noddi':
        acq_scheme, nmeas, model = setup_noddi()

    bvalues = np.loadtxt(join(path_diff, 'test/bvals.txt'))  # given in s/mm^2
    bvalues_SI = bvalues * 1e6  # now given in SI units as s/m^2
    gradient_directions = np.loadtxt(join(path_diff, 'test/bvecs.txt'))  # on the unit sphere

    # The delta and Delta times we know from the HCP documentation in seconds
    delta = 0.0106
    Delta = 0.0431

    # The acquisition scheme used in the toolbox is then created as follows:
    acq_scheme = acquisition_scheme_from_bvalues(bvalues_SI, gradient_directions, delta, Delta)

    ##-- Reading the DWI nifti image
    from dipy.io.image import load_nifti
    image_path = join(path_diff, "data.nii.gz")
    # image_path = join(path_diff, "data_snr10.nii.gz")
    data = nib.load(image_path).get_fdata()

    # plotting an axial slice
    # import matplotlib.pyplot as plt
    axial_middle = data.shape[2] // 2
    # plt.figure('Axial slice')
    # plt.imshow(data[:, :, axial_middle, 0].T, cmap='gray', origin='lower')
    # plt.show()

    data = data[:, :, axial_middle, :]

    ROIs = np.zeros_like(data[:, :, axial_middle])

    for idx, roi_im in enumerate(listdir(path_ROIs)):
        roi_img = nib.load((join(path_ROIs, roi_im))).get_fdata()
        roi_slice = roi_img[:, :, axial_middle]
        ROIs[roi_slice>0] = idx+1

    return acq_scheme, data, model, ROIs


def load_toy_phantom_ballstick(dimx, dimy, model, acq_scheme):
    # simulate a simple 10x10 image
    nvox = dimx * dimy

    Dpar_sim = 1.7e-9 + np.random.normal(0, 0.1e-9, nvox)
    Diso_sim = 2.5e-9 + np.random.normal(0, 0.1e-9, nvox)
    fpar_sim = 0.3 + np.random.normal(0, 0.025, nvox)
    stick_ori_sim = (np.pi / 2, 0) + np.random.normal(0, 0.11, (nvox, 2))
    parameter_vector = model.parameters_to_parameter_vector(
        C1Stick_1_mu=stick_ori_sim,
        C1Stick_1_lambda_par=Dpar_sim,
        G1Ball_1_lambda_iso=Diso_sim,
        partial_volume_0=fpar_sim,
        partial_volume_1=1 - fpar_sim)
    data = model.simulate_signal(acq_scheme, parameter_vector) + np.random.normal(0, .01, (nvox, acq_scheme.number_of_measurements))

    parameter_vector = model.parameter_vector_to_parameters(parameter_vector)

    # create mask with regional ROIs
    mask = np.ones(nvox)

    return parameter_vector, mask, data


def load_toy_phantom_noddi(dimx, dimy, model, acq_scheme):
    # simulate a simple 10x10 image
    nvox = dimx * dimy

    mu = (np.pi, 0)
    SD1WatsonDistributed_1_SD1Watson_1_odi = [0.2, 0.6, 0.7]
    SD1WatsonDistributed_1_partial_volume_0 = [0.8, 0.6, 0.9]
    partial_volume_0 = [0.1, 0.1, 0.9]
    partial_volume_1 = [1 - x for x in partial_volume_0]
    # lambda_par = 1.7e-9  # in m^2/s
    # lambda_iso = 3e-9  # in m^2/s
    # f_0 = 0.3
    # f_1 = 0.7
    parameter_vector = model.parameters_to_parameter_vector(
        SD1WatsonDistributed_1_SD1Watson_1_mu=mu,
        SD1WatsonDistributed_1_SD1Watson_1_odi=SD1WatsonDistributed_1_SD1Watson_1_odi,
        SD1WatsonDistributed_1_partial_volume_0=SD1WatsonDistributed_1_partial_volume_0,
        partial_volume_0=partial_volume_0,
        partial_volume_1=partial_volume_1)

    data = model.simulate_signal(acq_scheme, parameter_vector) + np.random.normal(0, .05, (nvox, acq_scheme.number_of_measurements))

    parameter_vector = model.parameter_vector_to_parameters(parameter_vector)

    # create mask with regional ROIs
    mask = np.ones(nvox)

    return parameter_vector, mask, data


def load_dynamic_phantom_noddi(dim, model, acq_scheme, lambda_par, lambda_iso):
    # load concentric circle phantom
    m0 = dynamic(dim, 1)

    # use the m0 image to create ballstick model params
    mu = np.zeros([dim * dim, 2])
    SD1WatsonDistributed_1_SD1Watson_1_odi = np.zeros(dim * dim)
    SD1WatsonDistributed_1_partial_volume_0 = np.zeros(dim * dim)
    partial_volume_0 = np.zeros(dim * dim)
    G1Ball_1_lambda_iso = np.zeros(dim * dim)
    SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par = np.zeros(dim * dim)
    mask = np.zeros(dim * dim)
    for i in range(np.unique(m0.flatten()).__len__()):
        idx = m0.flatten() == np.unique(m0.flatten())[i]
        mu[idx] = [.75 * np.pi * np.unique(m0.flatten())[i], .25 * np.pi * np.unique(m0.flatten())[i]]
        SD1WatsonDistributed_1_SD1Watson_1_odi[idx] = 0.5 * np.unique(m0.flatten())[i]  # [0.2, 0.6, 0.7]
        SD1WatsonDistributed_1_partial_volume_0[idx] = 0.7 * np.unique(m0.flatten())[i]  # [0.8, 0.6, 0.9]
        G1Ball_1_lambda_iso[idx] = lambda_iso  # * np.unique(m0.flatten())[i]
        SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par[idx] = lambda_par  # * np.unique(m0.flatten())[i]
        partial_volume_0[idx] = 0.9 * np.unique(m0.flatten())[i]  # [0.1, 0.1, 0.9]
        # create mask
        mask[idx] = i
    partial_volume_1 = [1 - x for x in partial_volume_0]

    # create mask
    #mask = m0 > 0
    #mask = mask.flatten()

    # simulate signal with SNR = 100
    parameter_vector = model.parameters_to_parameter_vector(
        SD1WatsonDistributed_1_SD1Watson_1_mu=mu,
        SD1WatsonDistributed_1_SD1Watson_1_odi=SD1WatsonDistributed_1_SD1Watson_1_odi,
        SD1WatsonDistributed_1_partial_volume_0=SD1WatsonDistributed_1_partial_volume_0,
        partial_volume_0=partial_volume_0,
        partial_volume_1=partial_volume_1,
        G1Ball_1_lambda_iso=G1Ball_1_lambda_iso,
        SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par=SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par)
    data = model.simulate_signal(acq_scheme, parameter_vector) + np.random.normal(0, .01, (dim*dim, acq_scheme.number_of_measurements))

    # convert parameter vector to dictionary
    parameter_vector = model.parameter_vector_to_parameters(parameter_vector)

    return parameter_vector, mask, data, m0


# save workspace variables; vars = dir()
def save_workspace(filename):
    print(filename)
    shelf = shelve.open(filename, "n")
    for key in globals():
        try:
            # print(key)
            shelf[key] = globals()[key]
        except:
            print('ERROR shelving: {0}'.format(key))
    shelf.close()


# load workspace variables
def load_workspace(filename):
    shelf = shelve.open(filename)
    print(shelf)
    for key in shelf:
        try:
            print(key)
            globals()[key] = shelf[key]
        except:
            print('ERROR shelving: {0}'.format(key))
    shelf.close()


def main():

    model_name = 'noddi'  # ballstick, noddi

    # set up data paths and model
    dim = 25
    lambda_par = 1.7e-9   # fix diffusivity (stick & zeppelin)
    lambda_iso = 2.8e-9   # fix diffusivity (ball)
    acq_scheme, nmeas, model_sub, model_ful = setup_noddi(lambda_par, lambda_iso)
    # parameter_vector, mask, data = load_toy_phantom_noddi(dimx, dimy, model, acq_scheme)
    parameter_vector, mask, data, m0 = load_dynamic_phantom_noddi(dim, model_ful, acq_scheme, lambda_par, lambda_iso)
    mask_roi = deepcopy(mask)  # (mask rois)
    mask_glob = deepcopy(mask)  # (mask global)
    mask_glob[mask_glob > 0] = 1
    mask_glob = mask_glob.astype('bool')

    # LSQ fitting
    lsq_fit = model_sub.fit(acq_scheme, data, mask=mask_glob)
    parameter_vector_lsq = deepcopy(lsq_fit.fitted_parameters)
    # add diffusivities (fixed for LSQ; free for Bayes -> add noise to initial LSQ parameters)
    # parameter_vector_lsq['G1Ball_1_lambda_iso'] = np.zeros(dim*dim)
    # parameter_vector_lsq['G1Ball_1_lambda_iso'][mask_glob] = lambda_iso + np.random.normal(0, lambda_iso/5, np.count_nonzero(mask_glob))
    # parameter_vector_lsq['G1Ball_1_lambda_iso'][parameter_vector_lsq['G1Ball_1_lambda_iso'] > 3e-9] = 1.5e-9
    # parameter_vector_lsq['G1Ball_1_lambda_iso'][parameter_vector_lsq['G1Ball_1_lambda_iso'] < .1e-9] = 1.5e-9
    # parameter_vector_lsq['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'] = np.zeros(dim*dim)
    # parameter_vector_lsq['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'][mask_glob] = lambda_par + np.random.normal(0, lambda_par/5, np.count_nonzero(mask_glob))
    # parameter_vector_lsq['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'][parameter_vector_lsq['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'] > 3e-9] = 1.5e-9
    # parameter_vector_lsq['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'][parameter_vector_lsq['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'] < .1e-9] = 1.5e-9
    nsteps = 1000
    burn_in = 500
    # model = deepcopy(model_ful)
    model = deepcopy(model_sub)

    # hierarchical Bayesian fitting
    proc_start = time.time()
    acceptance_rate, param_conv, parameter_vector_bayes, parameter_vector_init, likelihood_stored, w_stored \
        = fit_bayes.fit(model, acq_scheme, data, parameter_vector_lsq, mask_glob, nsteps, burn_in)
    compute_time(proc_start, time.time())

    parameter_vector_lsq['G1Ball_1_lambda_iso'] = np.zeros(dim * dim)
    parameter_vector_lsq['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'] = np.zeros(dim * dim)
    # revert LSQ diffusivities back to fixed values
    parameter_vector_lsq['G1Ball_1_lambda_iso'][mask_glob] = lambda_iso
    parameter_vector_lsq['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'][mask_glob] = lambda_par
    parameter_vector_bayes['G1Ball_1_lambda_iso'] = parameter_vector_lsq['G1Ball_1_lambda_iso']
    parameter_vector_bayes['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'] = parameter_vector_lsq['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par']

    # print: initialisation, correct value, mean (after burn-in) Bayes-fitted value
    nparams = np.sum(np.array(list(model.parameter_cardinality.values())))
    roi_vals = np.unique(mask)[np.unique(mask) > 0]  # list of unique integers that identify each ROI (ignore 0's)
    roi_nvox = [[xx for xx, x in enumerate(mask == roi_vals[roi]) if x].__len__() for roi in
                range(roi_vals.__len__())]  # number of voxels in each ROI
    to_remove = [roi for roi in range(roi_vals.__len__()) if
                 roi_nvox[roi] < 2 * nparams]  # indices of ROIs with too few voxels
    roi_vals = np.delete(roi_vals, to_remove)
    idx_roi = [xx for xx, x in enumerate(mask == roi_vals[0]) if x]
    vox0 = idx_roi[0]
    vox1 = idx_roi[1]
    vox2 = idx_roi[2]
    vox3 = idx_roi[3]

    # ------------------------------------------------------------------------------------------------------------------
    # filename = '/home/epowell/code/python/dmipy-bayesian/data/shepp_logan_' + str(dims[0]) + 'x' + str(dims[1]) \
    #            + '_snr25_nsteps' + str(nsteps) + '_burn' + str(burn_in) + '.db'
    # save_workspace(filename)

    # ------------------------------------------------------------------------------------------------------------------
    # plot parameter convergence
    plt.rcParams.update({'font.size': 22})
    lw = 5
    fig, axs = plt.subplots(2, 5)

    axs[0, 0].plot(range(nsteps), param_conv['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'][vox0, :], color='seagreen', linewidth=lw)
    axs[0, 0].set_ylabel("D (stick/zeppelin) [$\mu$m/ms]")
    axs[0, 0].set_xlabel("MCMC iteration")
    axs[0, 0].set_ylim([model.parameter_ranges['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'][0] * 1e-9,
                        model.parameter_ranges['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'][1] * 1e-9])
    make_square_axes(axs[0, 0])

    axs[0, 1].plot(range(nsteps), param_conv['G1Ball_1_lambda_iso'][vox1, :], color='steelblue', linewidth=lw)
    axs[0, 1].set_ylabel("D (ball) [$\mu$m/ms]")
    axs[0, 1].set_xlabel("MCMC iteration")
    axs[0, 1].set_ylim([model.parameter_ranges['G1Ball_1_lambda_iso'][0] * 1e-9,
                        model.parameter_ranges['G1Ball_1_lambda_iso'][1] * 1e-9])
    make_square_axes(axs[0, 1])

    axs[0, 2].plot(range(nsteps), param_conv['partial_volume_0'][vox2, :], color='indigo', linewidth=lw)
    axs[0, 2].set_ylabel("f (ball) [a.u.]")
    axs[0, 2].set_xlabel("MCMC iteration")
    axs[0, 2].set_ylim(model.parameter_ranges['partial_volume_0'])
    make_square_axes(axs[0, 2])

    axs[0, 3].plot(range(nsteps), param_conv['SD1WatsonDistributed_1_partial_volume_0'][vox2, :], color='red', linewidth=lw)
    axs[0, 3].set_ylabel("f (Watson stick) [a.u.]")
    axs[0, 3].set_xlabel("MCMC iteration")
    axs[0, 3].set_ylim(model.parameter_ranges['SD1WatsonDistributed_1_partial_volume_0'])
    make_square_axes(axs[0, 3])

    axs[0, 4].plot(range(nsteps), param_conv['SD1WatsonDistributed_1_SD1Watson_1_odi'][vox3, :], color='gold', linewidth=lw)
    axs[0, 4].set_ylabel("ODI")
    axs[0, 4].set_xlabel("MCMC iteration")
    axs[0, 4].set_ylim(model.parameter_ranges['SD1WatsonDistributed_1_SD1Watson_1_odi'][0])
    make_square_axes(axs[0, 4])

    # plot parameter distributions after burn-in period
    nbins = 15
    vals = param_conv['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'][vox0, burn_in:-1] * 1e9  # multiply by 1e9 so gaussian has same scaling
    (mu, sigma) = scipy.stats.norm.fit(vals)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axs[1, 0].hist(vals, bins=nbins, color='tab:gray', edgecolor='k', alpha=.4, density=True)
    axs[1, 0].plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='seagreen', linewidth=lw)
    axs[1, 0].set_ylabel("frequency density")
    axs[1, 0].set_xlabel("D (stickzeppelin) [$\mu$m/ms]")
    make_square_axes(axs[1, 0])

    vals = param_conv['G1Ball_1_lambda_iso'][vox1, burn_in:-1] * 1e9  # multiply by 1e9 so gaussian has same scaling
    (mu, sigma) = scipy.stats.norm.fit(vals)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axs[1, 1].hist(vals, bins=nbins, color='tab:gray', edgecolor='k', alpha=.4, density=True)
    axs[1, 1].plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='steelblue', linewidth=lw)
    axs[1, 1].set_ylabel("frequency density")
    axs[1, 1].set_xlabel("D (ball) [$\mu$m/ms]")
    make_square_axes(axs[1, 1])

    vals = param_conv['partial_volume_0'][vox2, burn_in:-1]
    (mu, sigma) = scipy.stats.norm.fit(vals)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axs[1, 2].hist(vals, bins=nbins, color='tab:gray', edgecolor='k', alpha=.4, density=True)
    axs[1, 2].plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='indigo', linewidth=lw)
    axs[1, 2].set_ylabel("frequency density")
    axs[1, 2].set_xlabel("f (ball) [a.u.]")
    make_square_axes(axs[1, 2])

    vals = param_conv['SD1WatsonDistributed_1_partial_volume_0'][vox2, burn_in:-1]
    (mu, sigma) = scipy.stats.norm.fit(vals)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axs[1, 3].hist(vals, bins=nbins, color='tab:gray', edgecolor='k', alpha=.4, density=True)
    axs[1, 3].plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='red', linewidth=lw)
    axs[1, 3].set_ylabel("frequency density")
    axs[1, 3].set_xlabel("f (Watson stick) [a.u.]")
    make_square_axes(axs[1, 3])

    vals = param_conv['SD1WatsonDistributed_1_SD1Watson_1_odi'][vox3, burn_in:-1]
    (mu, sigma) = scipy.stats.norm.fit(vals)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axs[1, 4].hist(vals, bins=nbins, color='tab:gray', edgecolor='k', alpha=.4, density=True)
    axs[1, 4].plot(x, scipy.stats.norm.pdf(x, mu, sigma), color='gold', linewidth=lw)
    axs[1, 4].set_ylabel("frequency density")
    axs[1, 4].set_xlabel("ODI")
    make_square_axes(axs[1, 4])

    # ------------------------------------------------------------------------------------------------------------------
    # plot acceptance rate
    fig, axs = plt.subplots(1, 2)
    axs[0].set_ylabel("Acceptance Rate")
    axs[0].plot(range(nsteps), acceptance_rate['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'][vox0], color='seagreen', linewidth=lw)
    axs[0].plot(range(nsteps), acceptance_rate['G1Ball_1_lambda_iso'][vox0, :], color='steelblue', linewidth=lw)
    axs[0].plot(range(nsteps), acceptance_rate['partial_volume_0'][vox0, :], color='indigo', linewidth=lw)
    axs[0].plot(range(nsteps), acceptance_rate['SD1WatsonDistributed_1_partial_volume_0'][vox0, :], color='red', linewidth=lw)
    axs[0].plot(range(nsteps), acceptance_rate['SD1WatsonDistributed_1_SD1Watson_1_odi'][vox0, :], color='gold', linewidth=lw)
    axs[0].legend(['Dpar', 'Diso', 'fball', 'fWatsonstick', 'ODI'])

    # plot likelihood
    axs[1].set_ylabel("Likelihood")
    axs[1].plot(range(nsteps), likelihood_stored['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'][vox0, :], color='seagreen', linewidth=lw)
    axs[1].plot(range(nsteps), likelihood_stored['G1Ball_1_lambda_iso'][vox0, :], color='steelblue', linewidth=lw)
    axs[1].plot(range(nsteps), likelihood_stored['partial_volume_0'][vox0, :], color='indigo', linewidth=lw)
    axs[1].plot(range(nsteps), likelihood_stored['SD1WatsonDistributed_1_partial_volume_0'][vox0, :], color='red', linewidth=lw)
    axs[1].plot(range(nsteps), likelihood_stored['SD1WatsonDistributed_1_SD1Watson_1_odi'][vox0, :], color='gold', linewidth=lw)
    axs[1].legend(['Dpar', 'Diso', 'fball', 'fWatsonstick', 'ODI'])

    # ------------------------------------------------------------------------------------------------------------------
    # plot maps: LSQ, Bayes, GT
    plt.rcParams.update({'font.size': 42})
    fig = plt.figure(figsize=(3, 5))
    grid = AxesGrid(fig, 111, nrows_ncols=(3, 5), axes_pad=0, cbar_mode='edge', cbar_location='bottom', cbar_pad=.25)
    cmap_D = copy(mpl.cm.BuPu_r)
    cmap_D.set_bad(color='k')
    cmap_f = copy(mpl.cm.OrRd_r)
    cmap_f.set_bad(color='k')
    cmap_mu = copy(mpl.cm.YlGn_r)
    cmap_mu.set_bad(color='k')
    clims_D = [1.5e-9, 3e-9]
    clims_f = [0, .75]
    clims_mu = [0, np.pi]
    # remove axes ticks and labels
    for g in range(8):
        grid[g].axes.set_xticklabels([])
        grid[g].axes.set_yticklabels([])
        grid[g].axes.set_xticks([])
        grid[g].axes.set_yticks([])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_D[0], vmax=clims_D[1]), cmap=cmap_D),
                        cax=grid[0].cax, orientation='horizontal', label='0-3 $\mu$m$^2$/ms')
    cbar.set_ticks([])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_D[0], vmax=clims_D[1]), cmap=cmap_D),
                        cax=grid[1].cax, orientation='horizontal', label='0-3 $\mu$m$^2$/ms')
    cbar.set_ticks([])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_f[0], vmax=clims_f[1]), cmap=cmap_f),
                        cax=grid[2].cax, orientation='horizontal', label='0-0.75 a.u.')
    cbar.set_ticks([])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=clims_mu[0], vmax=clims_mu[1]), cmap=cmap_mu),
        cax=grid[3].cax, orientation='horizontal', label='0-$\pi$ rad')
    cbar.set_ticks([])

    # transform to rotate brains by 90
    t = transforms.Affine2D().rotate_deg(90)

    # LSQ
    im = np.reshape(parameter_vector_lsq['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'], [dim,dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[0].imshow(im, vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)

    im = np.reshape(parameter_vector_lsq['G1Ball_1_lambda_iso'], [dim,dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[1].imshow(im, vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)

    im = np.reshape(parameter_vector_lsq['partial_volume_0'], [dim,dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[2].imshow(im, vmin=clims_f[0], vmax=clims_f[1], cmap=cmap_f)

    im = np.reshape(parameter_vector_lsq['SD1WatsonDistributed_1_partial_volume_0'], [dim, dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[3].imshow(im, vmin=clims_f[0], vmax=clims_f[1], cmap=cmap_f)

    im = np.reshape(parameter_vector_lsq['SD1WatsonDistributed_1_SD1Watson_1_odi'], [dim,dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[4].imshow(im, vmin=clims_mu[0], vmax=clims_mu[1], cmap=cmap_mu)

    # Bayes
    im = np.reshape(parameter_vector_bayes['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'], [dim,dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[5].imshow(im, vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)

    im = np.reshape(parameter_vector_bayes['G1Ball_1_lambda_iso'], [dim,dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[6].imshow(im, vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)

    im = np.reshape(parameter_vector_bayes['partial_volume_0'], [dim,dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[7].imshow(im, vmin=clims_f[0], vmax=clims_f[1], cmap=cmap_f)

    im = np.reshape(parameter_vector_bayes['SD1WatsonDistributed_1_partial_volume_0'], [dim, dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[8].imshow(im, vmin=clims_f[0], vmax=clims_f[1], cmap=cmap_f)

    im = np.reshape(parameter_vector_bayes['SD1WatsonDistributed_1_SD1Watson_1_odi'], [dim,dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[9].imshow(im, vmin=clims_mu[0], vmax=clims_mu[1], cmap=cmap_mu)

    # GT
    im = np.reshape(parameter_vector['SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par'], [dim, dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[10].imshow(im, vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)

    im = np.reshape(parameter_vector['G1Ball_1_lambda_iso'], [dim, dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[11].imshow(im, vmin=clims_D[0], vmax=clims_D[1], cmap=cmap_D)

    im = np.reshape(parameter_vector['partial_volume_0'], [dim, dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[12].imshow(im, vmin=clims_f[0], vmax=clims_f[1], cmap=cmap_f)

    im = np.reshape(parameter_vector['SD1WatsonDistributed_1_partial_volume_0'], [dim, dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[13].imshow(im, vmin=clims_f[0], vmax=clims_f[1], cmap=cmap_f)

    im = np.reshape(parameter_vector['SD1WatsonDistributed_1_SD1Watson_1_odi'], [dim, dim])
    im = scipy.ndimage.rotate(im, 90)
    im = np.ma.masked_where(im < 1e-15, im)
    grid[14].imshow(im, vmin=clims_mu[0], vmax=clims_mu[1], cmap=cmap_mu)

    grid[0].set_title('D (stick/zeppelin)')
    grid[1].set_title('D (ball)')
    grid[2].set_title('f (ball)')
    grid[3].set_title('f (Watson stick)')
    grid[4].set_title('ODI')
    grid[0].set_ylabel('LSQ', rotation=0, labelpad=50)
    grid[5].set_ylabel('Bayesian', rotation=0, labelpad=100)
    grid[10].set_ylabel('GT', rotation=0, labelpad=100)

if __name__ == '__main__':
    main()
