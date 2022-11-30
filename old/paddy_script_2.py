
# coding: utf-8

# In[1]:


#load modules
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import nibabel as nib
import dmipy

# In[2]:


#clone dmipy
#!git clone https://github.com/AthenaEPI/dmipy.git


# In[3]:


#change to the relevant directory
#os.chdir('dmipy')


# In[4]:


#install dmipy
#!pip install -e .


# In[5]:


# load the necessary dmipy modules
from dmipy.core import modeling_framework
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import BundleModel
from dmipy.core import modeling_framework
from dmipy.data import saved_acquisition_schemes
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues


# In[6]:


#set up the smt-noddi model
stick = cylinder_models.C1Stick()
zeppelin = gaussian_models.G2Zeppelin()

ball = gaussian_models.G1Ball()

bundle = BundleModel([stick, zeppelin])
bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
    'C1Stick_1_lambda_par','partial_volume_0')
bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)

smt_noddi_mod = modeling_framework.MultiCompartmentSphericalMeanModel(models=[bundle, ball])

#fix the isotropic diffusivity
smt_noddi_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)








# In[7]:


#set up the normal noddi model
from dmipy.distributions.distribute_models import SD1WatsonDistributed

watsonbundle = SD1WatsonDistributed(models=[stick, zeppelin])

watsonbundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
watsonbundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
watsonbundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)

noddi_mod = modeling_framework.MultiCompartmentModel(models=[watsonbundle, ball])

noddi_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)



#set up stick-zeppelin model
# sm_stickzep = modeling_framework.MultiCompartmentSphericalMeanModel(models=[stick, zeppelin])
# stickzep = modeling_framework.MultiCompartmentModel(models=[stick, zeppelin])


# In[8]:


#get the saved hcp scheme
acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()


# In[9]:


#choose some parameters for the simulated SMT-NODDI model 
#SD1WatsonDistributed_1_SD1Watson_1_odi = [0.2, 0.6, 0.7]
#SD1WatsonDistributed_1_partial_volume_0 = [0.8, 0.6, 0.9]

#limits on the parameters for wm/gm/csf
# stick_par_wm = [1e-9, 2e-9]
# stick_par_gm = [1e-9, 2e-9]
# stick_par_csf = [1e-9, 2e-9]
lambda_par_wm = [1.5e-9, 2.5e-9]
lambda_par_gm = [1.5e-9, 2.5e-9]
lambda_par_csf = [1.5e-9, 2.5e-9]
lambda_iso_wm = [2e-9, 3e-9]
lambda_iso_gm = [2e-9, 3e-9]
lambda_iso_csf = [2.8e-9, 3e-9]
odi_wm = [0.01, 0.3]
odi_gm = [0.6, 0.9]
odi_csf = [0, 1]
f_stick_wm = [0.7, 0.9]
f_stick_gm = [0.6, 8]
f_stick_csf = [0.8, 1]
f_bundle_wm = [0.7, 0.9]
f_bundle_gm = [0.8, 1]
f_bundle_csf = [0, 0.2]
# f_free_wm = 1-f_bundle_wm
# f_free_gm = 1-f_bundle_gm
# f_free_csf = 1-f_bundle_csf


#number of voxels for each tissue type
#n_wm = 40000
#n_gm = 20000
#n_csf = 10000 
n_wm = 500
n_gm = 300
n_csf = 100

ROImask = np.concatenate((np.ones((n_wm,)),2*np.ones((n_gm,)),3*np.ones((n_csf,))))                      

# stick_par = np.concatenate((np.random.uniform(low=stick_par_wm[0],high=stick_par_wm[1], size = n_wm),
#                       np.random.uniform(low=stick_par_gm[0],high=stick_par_gm[1], size = n_gm),
#                       np.random.uniform(low=stick_par_csf[0],high=stick_par_csf[1], size = n_csf)))

lambda_par = np.concatenate((np.random.uniform(low=lambda_par_wm[0],high=lambda_par_wm[1], size = n_wm),
                      np.random.uniform(low=lambda_par_gm[0],high=lambda_par_gm[1], size = n_gm),
                      np.random.uniform(low=lambda_par_csf[0],high=lambda_par_csf[1], size = n_csf)))

odi = np.concatenate((np.random.uniform(low=odi_wm[0],high=odi_wm[1], size = n_wm),
                      np.random.uniform(low=odi_gm[0],high=odi_gm[1], size = n_gm),
                      np.random.uniform(low=odi_csf[0],high=odi_csf[1], size = n_csf)))                

f_stick = np.concatenate((np.random.uniform(low=f_bundle_wm[0],high=f_bundle_wm[1], size = n_wm),
                      np.random.uniform(low=f_bundle_gm[0],high=f_bundle_gm[1], size = n_gm),
                      np.random.uniform(low=f_bundle_csf[0],high=f_bundle_csf[1], size = n_csf)))

f_bundle = np.concatenate((np.random.uniform(low=f_bundle_wm[0],high=f_bundle_wm[1], size = n_wm),
                      np.random.uniform(low=f_bundle_gm[0],high=f_bundle_gm[1], size = n_gm),
                      np.random.uniform(low=f_bundle_csf[0],high=f_bundle_csf[1], size = n_csf)))

lambda_iso = np.concatenate((np.random.uniform(low=lambda_iso_wm[0],high=lambda_iso_wm[1], size = n_wm),
                      np.random.uniform(low=lambda_iso_gm[0],high=lambda_iso_gm[1], size = n_gm),
                      np.random.uniform(low=lambda_iso_csf[0],high=lambda_iso_csf[1], size = n_csf)))

f_free = 1 - f_bundle


#images are a bit easier to deal with?
#n_wm+n_gm_n_csf
#np.reshape((np.sqrt(n_wm+n_gm_n_csf),np.sqrt()))

#put into a big parameter vector that can be passed into simulate_signal
# parameters_smt_noddi = smt_noddi_mod.parameters_to_parameter_vector(BundleModel_1_C1Stick_1_lambda_par=lambda_par,
#                                                                 BundleModel_1_G2Zeppelin_1_lambda_par=zep_par,
#                                                                 BundleModel_1_partial_volume_0=f_stick,
#                                                                 partial_volume_0=f_bundle,
#                                                                 partial_volume_1=f_free,
#                                                                 G1Ball_1_lambda_iso = lambda_iso)

# signals = smt_noddi_mod.simulate_signal(acq_scheme,parameters_smt_noddi)



# In[10]:


noddi_mod.parameter_names


# In[11]:


#choose some parameters for the simulated NORMAL NODDI model 
#SD1WatsonDistributed_1_SD1Watson_1_odi = [0.2, 0.6, 0.7]
#SD1WatsonDistributed_1_partial_volume_0 = [0.8, 0.6, 0.9]

#limits on the parameters for wm/gm/csf
# stick_par_wm = [1e-9, 2e-9]
# stick_par_gm = [1e-9, 2e-9]
# stick_par_csf = [1e-9, 2e-9]
lambda_par_wm = [1.5e-9, 2.5e-9]
lambda_par_gm = [1.5e-9, 2.5e-9]
lambda_par_csf = [1.5e-9, 2.5e-9]
odi_wm = [0.01, 0.3]
odi_gm = [0.6, 0.9]
odi_csf = [0, 1]
f_stick_wm = [0.7, 0.9]
f_stick_gm = [0.6, 8]
f_stick_csf = [0.8, 1]
f_bundle_wm = [0.7, 0.9]
f_bundle_gm = [0.8, 1]
f_bundle_csf = [0, 0.2]
# f_free_wm = 1-f_bundle_wm
# f_free_gm = 1-f_bundle_gm
# f_free_csf = 1-f_bundle_csf
mu_1 = [0, np.pi]
mu_2 = [-np.pi,np.pi]
lambda_iso_wm = [2e-9, 3e-9]
lambda_iso_gm = [2e-9, 3e-9]
lambda_iso_csf = [2.8e-9, 3e-9]
    
    

#number of voxels for each tissue type
#n_wm = 40000
#n_gm = 20000
#n_csf = 10000 
n_wm = 500
n_gm = 300
n_csf = 100

ROImask = np.concatenate((np.zeros((n_wm,)),np.ones((n_gm,)),2*np.ones((n_csf,))))                      

lambda_par = np.concatenate((np.random.uniform(low=lambda_par_wm[0],high=lambda_par_wm[1], size = n_wm),
                      np.random.uniform(low=lambda_par_gm[0],high=lambda_par_gm[1], size = n_gm),
                      np.random.uniform(low=lambda_par_csf[0],high=lambda_par_csf[1], size = n_csf)))

odi = np.concatenate((np.random.uniform(low=odi_wm[0],high=odi_wm[1], size = n_wm),
                      np.random.uniform(low=odi_gm[0],high=odi_gm[1], size = n_gm),
                      np.random.uniform(low=odi_csf[0],high=odi_csf[1], size = n_csf)))                

f_stick = np.concatenate((np.random.uniform(low=f_bundle_wm[0],high=f_bundle_wm[1], size = n_wm),
                      np.random.uniform(low=f_bundle_gm[0],high=f_bundle_gm[1], size = n_gm),
                      np.random.uniform(low=f_bundle_csf[0],high=f_bundle_csf[1], size = n_csf)))

f_bundle = np.concatenate((np.random.uniform(low=f_bundle_wm[0],high=f_bundle_wm[1], size = n_wm),
                      np.random.uniform(low=f_bundle_gm[0],high=f_bundle_gm[1], size = n_gm),
                      np.random.uniform(low=f_bundle_csf[0],high=f_bundle_csf[1], size = n_csf)))

lambda_iso = np.concatenate((np.random.uniform(low=lambda_iso_wm[0],high=lambda_iso_wm[1], size = n_wm),
                      np.random.uniform(low=lambda_iso_gm[0],high=lambda_iso_gm[1], size = n_gm),
                      np.random.uniform(low=lambda_iso_csf[0],high=lambda_iso_csf[1], size = n_csf)))


f_free = 1 - f_bundle

mu_1 = np.random.uniform(low=mu_1[0],high=mu_1[1],size=n_wm+n_gm+n_csf)
mu_2 = np.random.uniform(low=mu_2[0],high=mu_2[1],size=n_wm+n_gm+n_csf)

mu = np.stack((mu_1,mu_2),axis=1)





#images are a bit easier to deal with?
#n_wm+n_gm_n_csf
#np.reshape((np.sqrt(n_wm+n_gm_n_csf),np.sqrt()))

#put into a big parameter vector that can be passed into simulate_signal
# parameters_noddi = noddi_mod.parameters_to_parameter_vector(SD1WatsonDistributed_1_partial_volume_0=f_stick,
#                                                                 SD1WatsonDistributed_1_SD1Watson_1_mu=mu,
#                                                                 SD1WatsonDistributed_1_SD1Watson_1_odi=odi,
#                                                                 partial_volume_0=f_bundle,
#                                                                 partial_volume_1=f_free,
#                                                                 G1Ball_1_lambda_iso = lambda_iso,
#                                                                 SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par = lambda_par)

parameters_noddi = noddi_mod.parameters_to_parameter_vector(SD1WatsonDistributed_1_partial_volume_0=f_stick,
                                                                SD1WatsonDistributed_1_SD1Watson_1_mu=mu,
                                                                SD1WatsonDistributed_1_SD1Watson_1_odi=odi,
                                                                partial_volume_0=f_bundle,
                                                                partial_volume_1=f_free)



signals = noddi_mod.simulate_signal(acq_scheme,parameters_noddi)



# In[12]:


#add some noise
def add_noise(data, scale=0.02):
    data_real = data + np.random.normal(scale=scale, size=np.shape(data))
    data_imag = np.random.normal(scale=scale, size=np.shape(data))
    data_noisy = np.sqrt(data_real**2 + data_imag**2)

    return data_noisy


SNR = [10,25,50,75,100]
noisy_signals = {}
for i in range(0,len(SNR)):
    noisy_signals[SNR[i]] = add_noise(signals,scale=1/SNR[i])




# In[13]:


# os.mkdir('sims')
# #save
# #as numpy
# np.save('sims/noisy_signals_smt_noddi.npy',noisy_signals)
# np.save('sims/signals_smt_noddi.npy',signals)
# np.save('sims/ground_truth_smt_noddi_parameters.npy',parameters_smt_noddi)
# np.save('sims/ground_truth_ROImask.npy',ROImask)
# #as nifti
# for SNR in noisy_signals:
#     nib.save(nib.Nifti1Image(noisy_signals[SNR], np.eye(4)),'sims/SNR_' + str(SNR) + '_noisy_signals_smt_noddi.nii.gz')
# nib.save(nib.Nifti1Image(signals, np.eye(4)),'sims/signals_smt_noddi.nii.gz')
# nib.save(nib.Nifti1Image(ROImask, np.eye(4)),'sims/ground_truth_ROImask.nii.gz')



# In[14]:


# import shutil
# shutil.make_archive('sims', 'zip', 'sims')


# In[15]:


#do the lsq fit
#acq_scheme.spherical_mean_scheme.gradient_directions = np.zeros((np.shape(acq_scheme.bvalues)[0],3))
#acq_scheme.spherical_mean_scheme.b0_mask = 

sm_lsq_fit = smt_noddi_mod.fit(acq_scheme,noisy_signals[10])

#lsq_fit = noddi_mod.fit(acq_scheme,noisy_signals[10])



# In[16]:


fig, axs = plt.subplots(1, 5, figsize=[25, 5])

for roi in range(0,int(np.max(ROImask))+1):
    axs[0].plot(f_stick[ROImask==roi],sm_lsq_fit.fitted_parameters['BundleModel_1_partial_volume_0'][ROImask==roi],'o',markersize=1)
        
    axs[1].plot(f_bundle[ROImask==roi],sm_lsq_fit.fitted_parameters['partial_volume_0'][ROImask==roi],'o',markersize=1)
    
    #axs[2].plot(lambda_iso[ROImask==roi],lsq_fit.fitted_parameters['G1Ball_1_lambda_iso'][ROImask==roi],'o')

    #axs[2].plot(lambda_par[ROImask==roi],lsq_fit.fitted_parameters['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask==roi],'o')

#    axs[0].plot(f_stick[ROImask==roi],lsq_fit.fitted_parameters['BundleModel_1_partial_volume_0'][ROImask==roi],'o')


for i in range(0,2):
    axs[i].set_xlim([0,1])
    axs[i].set_ylim([0,1])
    


# In[17]:

grad_dirs = np.tile([1,1,1] / np.linalg.norm([1,1,1]), [np.unique(acq_scheme.bvalues).shape[0], 1])
acq_scheme_mod = acquisition_scheme_from_bvalues(np.unique(acq_scheme.bvalues), grad_dirs, acq_scheme.delta[0], acq_scheme.Delta[0])

#get the lsq fit ready for fit_bayes
parameter_vector_init = sm_lsq_fit.fitted_parameters
E_fit = smt_noddi_mod.simulate_signal(acq_scheme_mod, smt_noddi_mod.parameters_to_parameter_vector(**sm_lsq_fit.fitted_parameters))
E_fit = np.squeeze(E_fit)




# In[ ]:


import asyncio
from importlib import reload

import fit_bayes
from fit_bayes import fit, tform_params  # , dict_to_array, array_to_dict
fit_bayes = reload(fit_bayes)

nsteps=1000
burn_in=500

sm_noisy_signals = np.zeros((n_wm+n_gm+n_csf,acq_scheme.shell_bvalues.shape[0]))
for i in range(0,acq_scheme.shell_bvalues.shape[0]):
    idx = np.where(acq_scheme.shell_indices == i)
    tmp = np.squeeze(noisy_signals[100][:,idx])
    sm_noisy_signals[:,i] = np.mean(tmp,axis=1)
    

#reshape them
acceptance_rate, param_conv, parameter_vector_bayes, parameter_vector_init, likelihood_stored, w_stored = fit_bayes.fit(smt_noddi_mod, acq_scheme_mod, sm_noisy_signals, parameter_vector_init, ROImask+1, nsteps, burn_in)
    
# acceptance_rate_tmp, param_conv_tmp, parameter_vector_bayes_tmp, likelihood_stored_tmp, w_stored_tmp     = fit_bayes.fit(noddi_mod, acq_scheme, noisy_signals[10], E_fit, parameter_vector_init, ROImask, nsteps, burn_in)


# In[ ]:


fig, axs = plt.subplots(1, 5, figsize=[25, 5])

for roi in range(0,int(np.max(ROImask))+1):
    axs[0].plot(f_stick[ROImask==roi],parameter_vector_bayes['BundleModel_1_partial_volume_0'][ROImask==roi],'o',markersize=1)
        
    axs[1].plot(f_bundle[ROImask==roi],parameter_vector_bayes['partial_volume_0'][ROImask==roi],'o',markersize=1)
    
    #axs[2].plot(lambda_iso[ROImask==roi],lsq_fit.fitted_parameters['G1Ball_1_lambda_iso'][ROImask==roi],'o')

    #axs[2].plot(lambda_par[ROImask==roi],lsq_fit.fitted_parameters['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask==roi],'o')

#    axs[0].plot(f_stick[ROImask==roi],lsq_fit.fitted_parameters['BundleModel_1_partial_volume_0'][ROImask==roi],'o')


for i in range(0,2):
    axs[i].set_xlim([0,1])
    axs[i].set_ylim([0,1])
