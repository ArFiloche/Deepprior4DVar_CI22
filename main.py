import sys
sys.path.append('/Users/arthur_lip6/Projets_info/Deep_prior_CI2022')

import os
import shutil

import torch
torch.manual_seed(42)

import numpy as np
np.random.seed(42)

from dynamics import*
from utils import*

from _4DVar import*
from _DeepPrior4DVar import*

#### DIRECTORY MANAGEMENT ###

root_dir = './data/generated/'
save_dir = './data/estimated/'

if os.path.exists('./data'):
    shutil.rmtree('./data')
    
os.makedirs(root_dir)
os.makedirs('./data/results/')
os.makedirs(save_dir+'4DVar/')
os.makedirs(save_dir+'4DVar_reg/')
os.makedirs(save_dir+'4DVar_deep/')

#### GENERATE SYNTHETIC DATA ###

n_sample=100
dynamics=SW(dx,dy)
print('Generating synthetic data')
generator=DataGenerator(dynamics,T=10,subsample=3,sigma=0.05)
generator.dataset(n_sample, root_dir)

#### ASSIMILATION ####

results=np.zeros((n_sample,4,5)) # EPE, AE, ||grad||, ||div||, ||lap||

for i in range(n_sample):
    print(i)
    
    #load data
    initial_condition=np.load(root_dir+'initial_conditions/'+'{0:04}'.format(int(i))+'.npy')
    w0_truth=torch.Tensor(initial_condition[1:,:,:])
    
    # results
    results[i,0,0]=EPE(w0_truth, w0_truth) #0
    results[i,0,1]=Angular_error(w0_truth, w0_truth) #0
    results[i,0,2]=norm_gradw(w0_truth)
    results[i,0,3]=norm_divw(w0_truth)
    results[i,0,4]=norm_lapw(w0_truth)
    
    Obs=np.load(root_dir+'Obs/'+'{0:04}'.format(int(i))+'.npy')
    Obs=torch.Tensor(Obs)
    
    Rm1=torch.ones(Obs.shape)*(Obs!=0)
    Rm1=torch.Tensor(Rm1)
    
    ## 4D-Var #####
    assim = strong_4DVar(dynamics=dynamics,
                         optimizer=optim.LBFGS([torch.zeros(0)],lr=0.75, max_iter=250))
    assim.fit(Obs=Obs,Rm1=Rm1)
    
    w0_4dvar= assim.initial_condition[1:,:,:]
    np.save(save_dir+'4DVar/'+'{0:04}'.format(int(i))+'.npy', w0_4dvar)
    
        # results
    results[i,1,0]=EPE(w0_truth, w0_4dvar)
    results[i,1,1]=Angular_error(w0_truth, w0_4dvar)
    results[i,1,2]=norm_gradw(w0_4dvar)
    results[i,1,3]=norm_divw(w0_4dvar)
    results[i,1,4]=norm_lapw(w0_4dvar)
    
    ## Thikonov 4D-Var #####
    smoothreg= smooth_regul(alpha=5e3, beta=5e2,dx=dx,dy=dy)
    assim = strong_4DVar(dynamics=dynamics, regul=smoothreg,
                         optimizer=optim.LBFGS([torch.zeros(0)],lr=0.75, max_iter=250))
    assim.fit(Obs=Obs,Rm1=Rm1)
    
    w0_4dvar_reg=assim.initial_condition[1:,:,:]
    np.save(save_dir+'4DVar_reg/'+'{0:04}'.format(int(i))+'.npy', w0_4dvar_reg)
    
        # results
    results[i,2,0]=EPE(w0_truth, w0_4dvar_reg)
    results[i,2,1]=Angular_error(w0_truth, w0_4dvar_reg)
    results[i,2,2]=norm_gradw(w0_4dvar_reg)
    results[i,2,3]=norm_divw(w0_4dvar_reg)
    results[i,2,4]=norm_lapw(w0_4dvar_reg)
    
    ## Deep prior 4D-Var #####
    netG = Generator()
    netG.apply(weights_init)
    
    noise = torch.randn(netG.nz, 1, 1)
    
    assim=deeprior_strong_4DVar(generator=netG, dynamics=dynamics,
                               lr = 0.002, beta1 = 0.5, n_epoch=2000)
    assim.fit(noise, Obs, Rm1)
    
    w0_4dvar_deep=assim.initial_condition[1:,:,:]
    np.save(save_dir+'4DVar_deep/'+'{0:04}'.format(int(i))+'.npy', w0_4dvar_deep)
    
    # results
    results[i,3,0]=EPE(w0_truth, w0_4dvar_deep)
    results[i,3,1]=Angular_error(w0_truth, w0_4dvar_deep)
    results[i,3,2]=norm_gradw(w0_4dvar_deep)
    results[i,3,3]=norm_divw(w0_4dvar_deep)
    results[i,3,4]=norm_lapw(w0_4dvar_deep)
    
    np.save('./data/results/main.npy',results) 