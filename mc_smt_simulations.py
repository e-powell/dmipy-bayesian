#!/usr/bin/env python
# coding: utf-8

# In[1]:


#TO DO - PLAY AROUND WITH SMT PARAMETERS UNTIL YOU GET SENSIBLE FA/MD! CAN MATCH SMT MAX/MIN TO HCP DATA FIT

#write simulation funct#[noiseless, directional] signal and model and ROImask [dti], set random seedion that outputs ground truth 



#TO DO CALCULATE BIAS AND VARIANCE


#Paddy to do: 1. test-restest simulations; 2. "lesion" simulation
#Paddy. 3. rois based on clustering from lsq fit - can do kmeans etc, might need a bit of tuning but should be ok






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

#WM = 1, GM = 2, CSF =3
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


# lambda_par = np.concatenate((np.random.uniform(low=lambda_par_wm[0],high=lambda_par_wm[1], size = n_wm),
#                       np.random.uniform(low=lambda_par_gm[0],high=lambda_par_gm[1], size = n_gm),
#                       np.random.uniform(low=lambda_par_csf[0],high=lambda_par_csf[1], size = n_csf)))
              
# f_stick = np.concatenate((np.random.uniform(low=f_stick_wm[0],high=f_stick_wm[1], size = n_wm),
#                       np.random.uniform(low=f_stick_gm[0],high=f_stick_gm[1], size = n_gm),
#                       np.random.uniform(low=f_stick_csf[0],high=f_stick_csf[1], size = n_csf)))

lambda_par = np.concatenate((np.random.normal(loc=np.mean(lambda_par_wm),scale=.1e-9, size = n_wm),
                      np.random.normal(loc=np.mean(lambda_par_gm),scale=.1e-9, size = n_gm),
                      np.random.normal(loc=np.mean(lambda_par_csf),scale=.1e-9, size = n_csf)))
              
f_stick = np.concatenate((np.random.normal(loc=np.mean(f_stick_wm),scale=0.05, size = n_wm),
                      np.random.normal(loc=np.mean(f_stick_gm),scale=0.05, size = n_gm),
                      np.random.normal(loc=np.mean(f_stick_csf),scale=0.001, size = n_csf)))


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
                                                        







# In[78]:


plt.imshow(np.reshape(ROImask_gt,(30,30)))


# In[8]:


#get the saved hcp scheme
from dmipy.data import saved_acquisition_schemes

acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()


# In[13]:


#simulate the signal using the DIRECTIONAL model 

raw_signals = mc_mod.simulate_signal(acq_scheme,parameters_vector)

from useful_functions import add_noise

signals = add_noise(raw_signals,snr=25)

#also do a "retest" simulation - same underlying parameters but different noise
signals_retest = add_noise(raw_signals,snr=25)


# In[10]:


#calculate the ROImask using a diffusion tensor fit in dipy

#set up the dipy aquisition
from dipy.core.gradients import gradient_table
gtab = gradient_table(acq_scheme.bvalues, acq_scheme.gradient_directions)

#
import dipy.reconst.dti as dti

tenmodel = dti.TensorModel(gtab)

tenfit = tenmodel.fit(signals)


# In[11]:


#threshold md and fa to estimate the ROImask
md_thresh = 1e-9
fa_thresh = 0.8

ROImask = np.zeros_like(ROImask_gt)

#white matter - less than md threshold and higher than fa threshold
ROImask[(tenfit.md < md_thresh) & (tenfit.fa > fa_thresh)] = 1
#grey matter - less than md threshold and less than fa threshold
ROImask[(tenfit.md < md_thresh) & (tenfit.fa < fa_thresh)] = 2
#csf - higher than md threshold and lower than fa threshold
ROImask[(tenfit.md > md_thresh) & (tenfit.fa < fa_thresh)] = 3



# In[14]:


#fit the MC spherical mean model (CHECK AGAINST DIRECTION AVERAGING FIRST!)
mc_smt_mod_lsq_fit = mc_smt_mod.fit(acq_scheme, signals)

mc_smt_mod_lsq_fit_retest = mc_smt_mod.fit(acq_scheme, signals_retest)


# In[17]:


def estimate_ROIs_kmeans(parameters_vector,n_clusters):
    #cluster the voxels into ROIs using kmeans - could do this iteratively?
    from sklearn.cluster import KMeans
    import numpy as np
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(parameters_vector)

    ROImask = kmeans.labels_ + 1
    
    return ROImask


n_clusters=3

#do it for the test
ROImask = estimate_ROIs_kmeans(mc_smt_mod_lsq_fit.fitted_parameters_vector, n_clusters)
#do it for the retest
ROImask_retest = estimate_ROIs_kmeans(mc_smt_mod_lsq_fit_retest.fitted_parameters_vector, n_clusters)




# In[16]:


fig, axs = plt.subplots(1, 2, figsize=[10, 5])

for roi in range(0,int(np.max(ROImask))+1):
    axs[0].plot(f_stick[ROImask==roi],mc_smt_mod_lsq_fit.fitted_parameters['BundleModel_1_partial_volume_0'][ROImask==roi],'o',markersize=1)
        
    axs[1].plot(lambda_par[ROImask==roi],mc_smt_mod_lsq_fit.fitted_parameters['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask==roi],'o',markersize=1)    

    #calculate MSE
    
    
#axs[0].set_xlim([0,1])
#axs[0].set_ylim([0,1])


# In[18]: (BUG: ROI test and retest masks are different -> plotting didn't work because different shapes)


#test-retest plot 
fig, axs = plt.subplots(1, 2, figsize=[10, 5])

'''
for roi in range(0,int(np.max(ROImask))+1):
    axs[0].plot(mc_smt_mod_lsq_fit.fitted_parameters['BundleModel_1_partial_volume_0'][ROImask==roi],
                mc_smt_mod_lsq_fit_retest.fitted_parameters['BundleModel_1_partial_volume_0'][ROImask_retest==roi],'o',markersize=1)
        
    axs[1].plot(mc_smt_mod_lsq_fit.fitted_parameters['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask==roi],
                mc_smt_mod_lsq_fit_retest.fitted_parameters['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask_retest==roi],'o',markersize=1)    
'''
for roi in range(0,int(np.max(ROImask))+1):
    axs[0].plot(mc_smt_mod_lsq_fit.fitted_parameters['BundleModel_1_partial_volume_0'][ROImask==roi],
                mc_smt_mod_lsq_fit_retest.fitted_parameters['BundleModel_1_partial_volume_0'][ROImask==roi],'o',markersize=1)
        
    axs[1].plot(mc_smt_mod_lsq_fit.fitted_parameters['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask==roi],
                mc_smt_mod_lsq_fit_retest.fitted_parameters['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask==roi],'o',markersize=1)    

    
    


# In[21]:


from useful_functions import create_spherical_mean_scheme
from dmipy.utils import spherical_mean

acq_scheme_smt = create_spherical_mean_scheme(acq_scheme)

#signals_sm = np.asarray([spherical_mean.estimate_spherical_mean_multi_shell(signals, acq_scheme) ] 
signals_sm = np.asarray([spherical_mean.estimate_spherical_mean_multi_shell(signals[i,:], acq_scheme) for i in range(0,n_wm+n_gm+n_csf)])

signals_sm_retest = np.asarray([spherical_mean.estimate_spherical_mean_multi_shell(signals_retest[i,:], acq_scheme) for i in range(0,n_wm+n_gm+n_csf)])




# In[22]:


E_fit = mc_smt_mod.simulate_signal(acq_scheme_smt,mc_smt_mod_lsq_fit.fitted_parameters_vector)

E_fit_retest = mc_smt_mod.simulate_signal(acq_scheme_smt,mc_smt_mod_lsq_fit_retest.fitted_parameters_vector)


# In[23]:


import time 
import fit_bayes_new


nsteps=1000
burn_in=500
nupdates=20

# hierarchical Bayesian fitting
#proc_start = time.time()
parameters_bayes_dict, acceptance_rate, parameter_convergence, likelihood, weights    = fit_bayes_new.fit(mc_smt_mod, acq_scheme_smt, signals_sm, E_fit, mc_smt_mod_lsq_fit.fitted_parameters, ROImask, nsteps, burn_in, nupdates)
#compute_time(proc_start, time.time())

#retest
parameters_bayes_dict_retest, acceptance_rate_retest, parameter_convergence_retest, likelihood_retest, weights_retest    = fit_bayes_new.fit(mc_smt_mod, acq_scheme_smt, signals_sm_retest, E_fit_retest, mc_smt_mod_lsq_fit_retest.fitted_parameters, ROImask_retest, nsteps, burn_in, nupdates)



# In[25]:


fig, axs = plt.subplots(1, 2, figsize=[10, 5])

for roi in range(0,int(np.max(ROImask))+1):
    axs[0].plot(f_stick[ROImask==roi],parameters_bayes_dict['BundleModel_1_partial_volume_0'][ROImask==roi],'o',markersize=1)
        
    axs[1].plot(lambda_par[ROImask==roi],parameters_bayes_dict['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask==roi],'o',markersize=1)    
    
#    MSE['BundleModel_1_partial_volume_0'][roi] = 
#    MSE['BundleModel_1_G2Zeppelin_1_lambda_par'][roi] = 

#axs[0].set_xlim([0,1])
#axs[0].set_ylim([0,1])


# In[32]: (BUG: ROI test and retest masks are different -> plotting didn't work because different shapes)


#test-retest plots - plot lsq on the same plot
fig, axs = plt.subplots(2, 2, figsize=[10, 10])

'''
for roi in range(0,int(np.max(ROImask))+1):
    
    #Least squares
    axs[0,0].plot(mc_smt_mod_lsq_fit.fitted_parameters['BundleModel_1_partial_volume_0'][ROImask==roi],
                mc_smt_mod_lsq_fit_retest.fitted_parameters['BundleModel_1_partial_volume_0'][ROImask_retest==roi],'o',markersize=1)
        
    axs[0,1].plot(mc_smt_mod_lsq_fit.fitted_parameters['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask==roi],
                mc_smt_mod_lsq_fit_retest.fitted_parameters['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask_retest==roi],'o',markersize=1)    

    
    #Bayesian
    axs[1,0].plot(parameters_bayes_dict['BundleModel_1_partial_volume_0'][ROImask==roi],
                parameters_bayes_dict_retest['BundleModel_1_partial_volume_0'][ROImask_retest==roi],
                'o',markersize=1)
        
    axs[1,1].plot(parameters_bayes_dict['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask==roi],
                parameters_bayes_dict_retest['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask_retest==roi],
                'o',markersize=1)    
'''
for roi in range(0,int(np.max(ROImask))+1):
    
    #Least squares
    axs[0,0].plot(mc_smt_mod_lsq_fit.fitted_parameters['BundleModel_1_partial_volume_0'][ROImask==roi],
                mc_smt_mod_lsq_fit_retest.fitted_parameters['BundleModel_1_partial_volume_0'][ROImask==roi],'o',markersize=1)
        
    axs[0,1].plot(mc_smt_mod_lsq_fit.fitted_parameters['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask==roi],
                mc_smt_mod_lsq_fit_retest.fitted_parameters['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask==roi],'o',markersize=1)    

    
    #Bayesian
    axs[1,0].plot(parameters_bayes_dict['BundleModel_1_partial_volume_0'][ROImask==roi],
                parameters_bayes_dict_retest['BundleModel_1_partial_volume_0'][ROImask==roi],
                'o',markersize=1)
        
    axs[1,1].plot(parameters_bayes_dict['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask==roi],
                parameters_bayes_dict_retest['BundleModel_1_G2Zeppelin_1_lambda_par'][ROImask==roi],
                'o',markersize=1)    
 



# In[34]: 


import matplotlib.pyplot as plt
import numpy as np    
    
def bland_altman_plot(data1, data2, *args, **kwargs):
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2                   # Difference between data1 and data2
    md      = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    
    


# In[57]: (BUG: ROI test and retest masks are different -> plotting didn't work because different shapes)


#Bland-Altman plots

#test-retest plots 
fig, axs = plt.subplots(2, 2, figsize=[10, 10])

for roi in range(0,int(np.max(ROImask))+1):
    for param, i in zip(mc_smt_mod_lsq_fit.fitted_parameters.keys(), range(0,len(mc_smt_mod_lsq_fit.fitted_parameters))):
    
        #Least squares
        # mean = np.mean([mc_smt_mod_lsq_fit.fitted_parameters[param][ROImask==roi], mc_smt_mod_lsq_fit_retest.fitted_parameters[param][ROImask_retest==roi]],axis=0)
        mean = np.mean([mc_smt_mod_lsq_fit.fitted_parameters[param][ROImask==roi], mc_smt_mod_lsq_fit_retest.fitted_parameters[param][ROImask==roi]],axis=0)

        # diff = mc_smt_mod_lsq_fit.fitted_parameters[param][ROImask==roi] - mc_smt_mod_lsq_fit_retest.fitted_parameters[param][ROImask_retest==roi]
        diff = mc_smt_mod_lsq_fit.fitted_parameters[param][ROImask==roi] - mc_smt_mod_lsq_fit_retest.fitted_parameters[param][ROImask==roi]
        
        axs[i,0].plot(mean, diff, 'o', markersize=1)
                
#         #plot mean of the difference for each ROI separately 
#         axs[i,0].axhline(np.mean(diff), color='gray', linestyle='--')
#         axs[i,0].axhline(np.mean(diff)  + 1.96*np.std(diff, axis=0), color='gray', linestyle='--')
#         axs[i,0].axhline(np.mean(diff)  - 1.96*np.std(diff, axis=0), color='gray', linestyle='--')
        

        #Bayesian
        # mean = np.mean([parameters_bayes_dict[param][ROImask==roi], parameters_bayes_dict_retest[param][ROImask_retest==roi]],axis=0)
        mean = np.mean([parameters_bayes_dict[param][ROImask==roi], parameters_bayes_dict_retest[param][ROImask==roi]],axis=0)

        # diff = parameters_bayes_dict[param][ROImask==roi] - parameters_bayes_dict_retest[param][ROImask_retest==roi]
        diff = parameters_bayes_dict[param][ROImask==roi] - parameters_bayes_dict_retest[param][ROImask==roi]
                    
        axs[i,1].plot(mean, diff, 'o', markersize=1)
        
        #for easy comparison, make the bayesian y-axis limits the same as the LSQ        
        axs[i,1].set_ylim(axs[i,0].get_ylim())

        
#         #plot mean of the difference for each ROI separately 
#         axs[i,1].axhline(np.mean(diff), color='gray', linestyle='--')
#         axs[i,1].axhline(np.mean(diff)  + 1.96*np.std(diff, axis=0), color='gray', linestyle='--')
#         axs[i,1].axhline(np.mean(diff)  - 1.96*np.std(diff, axis=0), color='gray', linestyle='--')


#add labels
axs[0,0].set_title('Least Squares')
axs[0,1].set_title('Bayesian')

for param, i in zip(mc_smt_mod_lsq_fit.fitted_parameters.keys(), range(0,len(mc_smt_mod_lsq_fit.fitted_parameters))):
    axs[i,0].set_ylabel(param)
    axs[i,0].set_ylabel(param)
    


#plot mean and sd of the difference for all voxels together 

#LSQ
# mean = np.mean([mc_smt_mod_lsq_fit.fitted_parameters[param],
#             mc_smt_mod_lsq_fit_retest.fitted_parameters[param]],axis=0)

for param, i in zip(mc_smt_mod_lsq_fit.fitted_parameters.keys(), range(0,len(mc_smt_mod_lsq_fit.fitted_parameters))):
    diff = mc_smt_mod_lsq_fit.fitted_parameters[param] - mc_smt_mod_lsq_fit_retest.fitted_parameters[param]

    axs[i,0].axhline(np.mean(diff), color='gray', linestyle='--')
    axs[i,0].axhline(np.mean(diff)  + 1.96*np.std(diff, axis=0), color='gray', linestyle='--')
    axs[i,0].axhline(np.mean(diff)  - 1.96*np.std(diff, axis=0), color='gray', linestyle='--')

    #Bayesian
    diff = parameters_bayes_dict[param] -                 parameters_bayes_dict_retest[param]        
    axs[i,1].axhline(np.mean(diff), color='gray', linestyle='--')
    axs[i,1].axhline(np.mean(diff)  + 1.96*np.std(diff, axis=0), color='gray', linestyle='--')
    axs[i,1].axhline(np.mean(diff)  - 1.96*np.std(diff, axis=0), color='gray', linestyle='--')



# #create Bland-Altman plot                  
# f, ax = plt.subplots(1, figsize = (8,5))
# sm.graphics.mean_diff_plot(df.A, df.B, ax = ax)

# #display Bland-Altman plot
# plt.show()


# In[26]:


plt.plot(parameter_convergence['BundleModel_1_G2Zeppelin_1_lambda_par'][300,:])


# In[100]:


#calculate the contrast to noise ratios, remember WM = 1, GM = 2, CSF =3

def calculate_CNR(ROI1,ROI2):
    mean_ROI1 = np.mean(ROI1)
    mean_ROI2 = np.mean(ROI2)
    
    var_ROI1 = np.var(ROI1)
    var_ROI2 = np.var(ROI2)
    
    CNR = np.abs(mean_ROI1 - mean_ROI2)/np.sqrt(var_ROI1 + var_ROI2)
    
    return CNR

CNR_LSQ={}
CNR_Bayes={}
    
for param, i in zip(mc_smt_mod_lsq_fit.fitted_parameters.keys(), range(0,len(mc_smt_mod_lsq_fit.fitted_parameters))):
    #LSQ
    
#     mean_WM = np.mean(mc_smt_mod_lsq_fit.fitted_parameters[param][ROImask==1])
#     mean_GM = np.mean(mc_smt_mod_lsq_fit.fitted_parameters[param][ROImask==2])
#     mean_CSF = np.mean(mc_smt_mod_lsq_fit.fitted_parameters[param][ROImask==3])
    
#     sd_WM = np.std(mc_smt_mod_lsq_fit.fitted_parameters[param][ROImask==1])
#     sd_GM = np.std(mc_smt_mod_lsq_fit.fitted_parameters[param][ROImask==2])
#     sd_CSF = np.std(mc_smt_mod_lsq_fit.fitted_parameters[param][ROImask==3])
    

    CNR_LSQ[param] = np.zeros(len(mc_smt_mod_lsq_fit.fitted_parameters)+1)
    CNR_Bayes[param] = np.zeros(len(mc_smt_mod_lsq_fit.fitted_parameters)+1)

    
    #between white and grey matter 
    CNR_LSQ[param][0] = calculate_CNR(mc_smt_mod_lsq_fit.fitted_parameters[param][ROImask==1],                                     mc_smt_mod_lsq_fit.fitted_parameters[param][ROImask==2])

    #between white matter and CSF
    CNR_LSQ[param][1] = calculate_CNR(mc_smt_mod_lsq_fit.fitted_parameters[param][ROImask==1],                                     mc_smt_mod_lsq_fit.fitted_parameters[param][ROImask==3])
 
    #between grey matter and CSF
    CNR_LSQ[param][2] = calculate_CNR(mc_smt_mod_lsq_fit.fitted_parameters[param][ROImask==2],                                     mc_smt_mod_lsq_fit.fitted_parameters[param][ROImask==3])
 

    #Bayesian
    
    #between white and grey matter 
    CNR_Bayes[param][0] = calculate_CNR(parameters_bayes_dict[param][ROImask==1],
                                     parameters_bayes_dict[param][ROImask==2])

    #between white matter and CSF
    CNR_Bayes[param][1] = calculate_CNR(parameters_bayes_dict[param][ROImask==1],
                                     parameters_bayes_dict[param][ROImask==3])
 
    #between grey matter and CSF
    CNR_Bayes[param][2] = calculate_CNR(parameters_bayes_dict[param][ROImask==2],
                                     parameters_bayes_dict[param][ROImask==3])

    
    


# In[162]:


fig, axs = plt.subplots(1, 2, figsize=[12, 5])

bar_width = 0.35

nbars = 3

LSQ_bars = np.arange(3)
Bayes_bars = [x + bar_width for x in LSQ_bars]

for param, i in zip(mc_smt_mod_lsq_fit.fitted_parameters.keys(), range(0,len(mc_smt_mod_lsq_fit.fitted_parameters))):
    axs[i].bar(LSQ_bars,CNR_LSQ[param],width=bar_width, label='Least Squares')
    axs[i].bar(Bayes_bars,CNR_Bayes[param],width=bar_width, label='Bayesian')

    axs[i].legend()

    axs[i].set_ylabel('Contrast-to-Noise Ratio')

    #axs[i].set_xticklabels('')
    axs[i].set_xticks([r + bar_width/2 for r in range(nbars)])
    axs[i].set_xticklabels(['WM:GM','WM:CSF','GM:CSF'])
    
    axs[i].set_title(param)

#['WM:GM','WM:CSF','GM:CSF']

#plt.bar(np.arange(3),data)


# In[156]:


[r + bar_width/2 for r in range(nbars)]


# In[95]:


mc_smt_mod_lsq_fit.fitted_parameters['BundleModel_1_G2Zeppelin_1_lambda_par']


# In[171]:


#make the ground truth parameters into an image - can just do this last!
np.sum(ROImask_gt==1)

np.sqrt(300)

fig, axs = plt.subplots(2, 2, figsize=[10, 12])

image_shape = (36,25)

for param, i in zip(mc_smt_mod_lsq_fit.fitted_parameters.keys(), range(0,len(mc_smt_mod_lsq_fit.fitted_parameters))):
#    axs[0,1].imshow(np.reshape(ROImask_gt,image_shape))

    #LSQ
    im=axs[i,0].imshow(np.reshape(mc_smt_mod_lsq_fit.fitted_parameters[param],image_shape))
    fig.colorbar(im,ax=axs[i,0])

    #Bayesian
    im=axs[i,1].imshow(np.reshape(parameters_bayes_dict[param],image_shape))
    fig.colorbar(im,ax=axs[i,1])


    
    
#add labels
axs[0,0].set_title('Least Squares')
axs[0,1].set_title('Bayesian')

for param, i in zip(mc_smt_mod_lsq_fit.fitted_parameters.keys(), range(0,len(mc_smt_mod_lsq_fit.fitted_parameters))):
    axs[i,0].set_ylabel(param)
    axs[i,0].set_ylabel(param)
    


# In[ ]:





# In[ ]:


#TO DO CALCULATE BIAS AND VARIANCE


#Paddy to do: 1. test-restest simulations; 2. "lesion" simulation
#Paddy. 3. rois based on clustering from lsq fit - can do kmeans etc, might need a bit of tuning but should be ok


# In[ ]:




