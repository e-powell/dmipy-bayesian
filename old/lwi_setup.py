#luminal water imaging example

# load some modules
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.stats
import shelve
import os
from datetime import datetime

# for fitting
from dmipy.core import modeling_framework
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models import cylinder_models, gaussian_models


# -------------------------------------------------------------------------------------------------------------------- #
# FUNCTIONS


# for biexponential model fitting (just with ball ball for now)
def biexpt2(TE,fa,T2a,T2b):
    return fa*np.exp(-TE/T2a) + (1-fa)*np.exp(-TE/T2b)


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


# -------------------------------------------------------------------------------------------------------------------- #
# BALL BALL MODEL FITTING
# load an example LWI image & plot
datadir = '/home/epowell/data/lwi/'
img = nib.load(datadir + 'inn104rwb.nii')
img = img.get_data()
mask = nib.load(datadir + 'inn104rwb_mask_pjs.nii.gz')
mask = mask.get_data()

#plt.imshow(img[:, :, 2, 0])
#plt.imshow(mask[:, :, 3])

# load the acquisition parameters & put them into dmipy format
gradecho = np.loadtxt(datadir + 'LWI_gradechoinv.txt', delimiter=",")
TE = gradecho[:, 4] * 10e6                              # originals are ms - so these are nanoseconds?
TE = TE - np.min(TE)                                    # normalise the TEs to 0
bvecs = gradecho[:, 0:3]
acq_scheme = acquisition_scheme_from_bvalues(TE,bvecs)  # put b-vals as TE so can fit R2 with ball ball model
normvox = img[100, 100, 3, :] / img[100, 100, 3, 0]

# define other parameters
T2min, T2max = 0.001, 0.5
T2grid = np.linspace(T2min, T2max, num=1000)
R2min, R2max = 0.001, 0.5                       # should be in nanoseconds I think (for correct dmipy scaling)
scaling = 1e-09
R2grid = np.linspace(R2min * scaling, R2max * scaling, num=1000)
f = 0.3
mu1 = 1e-11
sigma1 = 1e-12
mu2 = 1e-10
sigma2 = 3e-12

#
ball1 = gaussian_models.G1BallNormalDist()
ball2 = gaussian_models.G1BallNormalDist()
ballball = MultiCompartmentModel([ball1, ball2])
#ballballfit = ballball.fit(acq_scheme, normvox, Ns=100)
parameter_vector = ballball.parameters_to_parameter_vector(G1BallNormalDist_1_lambda_iso_mean=mu1,
                                                           G1BallNormalDist_2_lambda_iso_mean=mu2,
                                                           G1BallNormalDist_1_lambda_iso_std=sigma1,
                                                           G1BallNormalDist_2_lambda_iso_std=sigma2,
                                                           partial_volume_0=0.7,
                                                           partial_volume_1=0.3)
S = ballball.simulate_signal(acq_scheme, parameter_vector)

# try and get the ballnormaldist2 working!
balls = gaussian_models.G2BallNormalDist()
balls = MultiCompartmentModel([balls])
parameter_vector = balls.parameters_to_parameter_vector(G2BallNormalDist_1_lambda_iso_mean_0=mu1,
                                                        G2BallNormalDist_1_lambda_iso_mean_1=mu2,
                                                        G2BallNormalDist_1_lambda_iso_std_0=sigma1,
                                                        G2BallNormalDist_1_lambda_iso_std_1=sigma2,
                                                        G2BallNormalDist_1_dist_height=0.7)
S = balls.simulate_signal(acq_scheme, parameter_vector)
plt.plot(S)
# must set initial guesses, as brute-to-fine init. fails (tries to create a ~72GiB array [100x100x100x100x100])
balls.set_initial_guess_parameter('G2BallNormalDist_1_lambda_iso_mean_0', mu1)
balls.set_initial_guess_parameter('G2BallNormalDist_1_lambda_iso_mean_1', mu2)
balls.set_initial_guess_parameter('G2BallNormalDist_1_lambda_iso_std_0', sigma1)
balls.set_initial_guess_parameter('G2BallNormalDist_1_lambda_iso_std_1', sigma2)
balls.set_initial_guess_parameter('G2BallNormalDist_1_dist_height', 0.7)

ballsfit_vox = balls.fit(acq_scheme, normvox, Ns=100)

ballsfit_img = balls.fit(acq_scheme, img[:,:,2], mask=mask[:,:,2])
plt.imshow(ballsfit_img.fitted_parameters['G2BallNormalDist_1_lambda_iso_mean_1'])
plt.colorbar()
# now = datetime.now()
# save_workspace(os.getcwd() + '/lwi_' + now.strftime("%Y-%m-%d_%H-%M-%S") + '.db')
