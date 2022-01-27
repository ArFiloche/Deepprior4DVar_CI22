import os
import shutil

import torch
#torch.manual_seed(42)

import numpy as np
#np.random.seed(42)

# --------------- Physical prameters ---------------
L_x = 1E+6              # Length of domain in x-direction
L_y = 1E+6              # Length of domain in y-direction
g = 9.81                # Acceleration of gravity [m/s^2]
H = 100                 # Depth of fluid [m]

# --------------- Computational prameters ---------------
N_x = 64                            # Number of grid points in x-direction
N_y = 64                            # Number of grid points in y-direction
dx = L_x/(N_x - 1)                  # Grid spacing in x-direction
dy = L_y/(N_y - 1)

x = torch.linspace(-L_x/2, L_x/2, N_x)  # Array with x-points
y = torch.linspace(-L_y/2, L_y/2, N_y)  # Array with y-points
Xx, Yy = torch.meshgrid(x, y) 

class DataGenerator():
    
    def __init__(self,dynamics,T=10,subsample=3,sigma=0.1):
        
        self.T=T
        self.subsample=subsample
        self.sigma=sigma
        self.dynamics=dynamics
        
    def sample(self):
        
        # --------------- random initial conditions ---------------
        xx=np.random.uniform(3,7,1)*np.random.choice([-1,1], size=1, p=[0.5, 0.5])
        yy=np.random.uniform(3,7,1)*np.random.choice([-1,1], size=1, p=[0.5, 0.5])
        s=np.random.uniform(2,10,1)*1E+4
        
        X0=torch.zeros((3,N_x,N_y))
        X0[0,:,:]=5*torch.exp(-((Xx-L_x/xx)**2/(2*(s)**2) + (Yy-L_y/yy)**2/(2*(s)**2)))
        
        # first integration run to reach equilibrium
        for t in range(1000):
            X0=self.dynamics.forward(X0)
            
        initial_condition = X0.numpy()    
        #eta0_truth=X0[0,:,:]
        #w0_truth=X0[1:,:,:]
        
        # second run - final trajectory
        X = X0
        Obs = torch.zeros((self.T,X0.shape[0],X0.shape[1],X0.shape[2]))
        
        for t in range(self.T):
    
            if t%self.subsample==0:
                noise=torch.normal(0,self.sigma,X.shape)
                Obs[t,0,:,:] = X[0,:,:]+noise[0,:,:]
                
            X=self.dynamics.forward(X)
             
        return initial_condition, Obs
        
        
    def dataset(self, n_sample, dir_save):
        
        # Directories management         
        if os.path.exists(dir_save+'Obs'):
            shutil.rmtree(dir_save+'Obs')
        os.makedirs(dir_save+'Obs')

        if os.path.exists(dir_save+'initial_conditions'):
            shutil.rmtree(dir_save+'initial_conditions')
        os.makedirs(dir_save+'initial_conditions')
        
        for i in range(n_sample):
            initial_condition, Obs = self.sample()
            
            # save     
            np.save(dir_save+'initial_conditions/'+'{0:04}'.format(int(i)),
                    initial_condition)
            np.save(dir_save+'Obs/'+'{0:04}'.format(int(i)),
                    Obs)