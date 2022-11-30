#!/usr/bin/env python
# coding: utf-8

# In[1]:


#TO DO - PLAY AROUND WITH SMT PARAMETERS UNTIL YOU GET SENSIBLE FA/MD! CAN MATCH SMT MAX/MIN TO HCP DATA FIT

#write simulation funct#[noiseless, directional] signal and model and ROImask [dti], set random seedion that outputs ground truth 


# In[2]:


#load modules

import numpy as np
import matplotlib.pyplot as plt


# In[3]:


#set the random seed so that you get the same simulations
np.random.seed(seed=1)


# In[4]:


from dmipy.signal_models import cylinder_models, gaussian_models
stick = cylinder_models.C1Stick()
zeppelin = gaussian_models.G2Zeppelin()


# In[5]:


from dmipy.distributions.distribute_models import BundleModel
bundle = BundleModel([stick, zeppelin])
bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
    'C1Stick_1_lambda_par','partial_volume_0')
bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
bundle.set_equal_parameter('G2Zeppelin_1_mu', 'C1Stick_1_mu')


# In[6]:


#set up two models - directional and direction averaged

from dmipy.core import modeling_framework
mc_smt_mod = modeling_framework.MultiCompartmentSphericalMeanModel(
    models=[bundle])
mc_smt_mod.parameter_names


mc_mod = modeling_framework.MultiCompartmentModel(
    models=[bundle])
mc_mod.parameter_names


# In[7]:


#set up the ground truth parameter values in each ROI



#number of voxels for each tissue type
#n_wm = 40000
#n_gm = 20000
#n_csf = 10000 
n_wm = 500
n_gm = 300
n_csf = 100

ROImask_gt = np.concatenate((np.ones((n_wm,)),2*np.ones((n_gm,)),3*np.ones((n_csf,))))                      

#choose some parameters for the simulated SMT-NODDI model 
lambda_par_wm = [1.5e-9, 2.5e-9]
lambda_par_gm = [0.5e-9, 1.5e-9]
lambda_par_csf = [2.9e-9, 3e-9]
mu_1 = [0, np.pi]
mu_2 = [-np.pi,np.pi]
f_stick_wm = [0.6, 0.8]
f_stick_gm = [0.2, 0.4]
f_stick_csf = [0, 0.01]


lambda_par = np.concatenate((np.random.uniform(low=lambda_par_wm[0],high=lambda_par_wm[1], size = n_wm),
                      np.random.uniform(low=lambda_par_gm[0],high=lambda_par_gm[1], size = n_gm),
                      np.random.uniform(low=lambda_par_csf[0],high=lambda_par_csf[1], size = n_csf)))
              
f_stick = np.concatenate((np.random.uniform(low=f_stick_wm[0],high=f_stick_wm[1], size = n_wm),
                      np.random.uniform(low=f_stick_gm[0],high=f_stick_gm[1], size = n_gm),
                      np.random.uniform(low=f_stick_csf[0],high=f_stick_csf[1], size = n_csf)))

mu_1 = np.random.uniform(low=mu_1[0],high=mu_1[1],size=n_wm+n_gm+n_csf)
mu_2 = np.random.uniform(low=mu_2[0],high=mu_2[1],size=n_wm+n_gm+n_csf)

mu = np.stack((mu_1,mu_2),axis=1)

f_free = 1 - f_stick



# stick_par = np.concatenate((np.random.uniform(low=stick_par_wm[0],high=stick_par_wm[1], size = n_wm),
#                       np.random.uniform(low=stick_par_gm[0],high=stick_par_gm[1], size = n_gm),
#                       np.random.uniform(low=stick_par_csf[0],high=stick_par_csf[1], size = n_csf)))




#put into a big parameter vector that can be passed into simulate_signal
parameters_vector = mc_mod.parameters_to_parameter_vector(BundleModel_1_G2Zeppelin_1_mu=mu,
                                                                BundleModel_1_G2Zeppelin_1_lambda_par=lambda_par,
                                                                BundleModel_1_partial_volume_0=f_stick)
                                                        







# In[8]:


#get the saved hcp scheme
from dmipy.data import saved_acquisition_schemes

acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()


# In[9]:


#simulate the signal using the DIRECTIONAL model 

signals = mc_mod.simulate_signal(acq_scheme,parameters_vector)

from useful_functions import add_noise

signals = add_noise(signals,snr=50)


# In[10]:


#calculate the ROImask using dipy

#set up the dipy aquisition
from dipy.core.gradients import gradient_table
gtab = gradient_table(acq_scheme.bvalues, acq_scheme.gradient_directions)

#
import dipy.reconst.dti as dti

tenmodel = dti.TensorModel(gtab)

tenfit = tenmodel.fit(signals)


# In[11]:


plt.plot(tenfit.fa)


# In[12]:


plt.plot(tenfit.md)


# In[13]:


#threshold md and fa to estimate the ROImask
md_thresh = 1.5e-9
fa_thresh = 0.8

ROImask = np.zeros_like(ROImask_gt)

#white matter - less than md threshold and higher than fa threshold
ROImask[(tenfit.md < md_thresh) & (tenfit.fa > fa_thresh)] = 1
#grey matter - less than md threshold and less than fa threshold
ROImask[(tenfit.md < md_thresh) & (tenfit.fa < fa_thresh)] = 2
#csf - higher than md threshold and lower than fa threshold
ROImask[(tenfit.md > md_thresh) & (tenfit.fa < fa_thresh)] = 3



# In[16]:


#fit the MC spherical mean model (CHECK AGAINST DIRECTION AVERAGING FIRST!)
mc_smt_mod_lsq_fit = mc_smt_mod.fit(acq_scheme, signals)


# In[17]:


fig, axs = plt.subplots(1, 2, figsize=[10, 5])

for roi in range(0,int(np.max(ROImask))+1):
    axs[0].plot(f_stick[ROImask==roi],mc_smt_mod_lsq_fit.fitted_parameters['BundleModel_1_partial_volume_0'][ROImask==roi],'o',markersize=1)
        
    axs[1].plot(lambda_par[ROImask==roi],mc_smt_mod_lsq_fit.fitted_parameters['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask==roi],'o',markersize=1)    


#axs[0].set_xlim([0,1])
#axs[0].set_ylim([0,1])


# In[18]:


from useful_functions import create_spherical_mean_scheme
from dmipy.utils import spherical_mean

acq_scheme_smt = create_spherical_mean_scheme(acq_scheme)

#signals_sm = np.asarray([spherical_mean.estimate_spherical_mean_multi_shell(signals, acq_scheme) ] 
signals_sm = np.asarray([spherical_mean.estimate_spherical_mean_multi_shell(signals[i,:], acq_scheme) for i in range(0,n_wm+n_gm+n_csf)])




# In[19]:


E_fit = mc_smt_mod.simulate_signal(acq_scheme_smt,mc_smt_mod_lsq_fit.fitted_parameters_vector)


# In[20]:


import time 
import fit_bayes_new


nsteps=1000
burn_in=500
nupdates=20

# hierarchical Bayesian fitting
#proc_start = time.time()
parameters_bayes_dict, acceptance_rate, parameter_convergence, likelihood, weights    = fit_bayes_new.fit(mc_smt_mod, acq_scheme_smt, signals_sm, E_fit, mc_smt_mod_lsq_fit.fitted_parameters, ROImask, nsteps, burn_in, nupdates)
#compute_time(proc_start, time.time())





# In[22]:


fig, axs = plt.subplots(1, 2, figsize=[10, 5])

for roi in range(0,int(np.max(ROImask))+1):
    axs[0].plot(f_stick[ROImask==roi],parameters_bayes_dict['BundleModel_1_partial_volume_0'][ROImask==roi],'o',markersize=1)
        
    axs[1].plot(lambda_par[ROImask==roi],parameters_bayes_dict['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask==roi],'o',markersize=1)    


#axs[0].set_xlim([0,1])
#axs[0].set_ylim([0,1])


# In[24]:


plt.plot(parameter_convergence['BundleModel_1_G2Zeppelin_1_lambda_par'][600,:])


# In[ ]:




