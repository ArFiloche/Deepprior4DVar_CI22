import torch.nn as nn
import torch
import torch.optim as optim

class deeprior_strong_4DVar():
    
    def __init__(self, generator, dynamics,
                 lr = 0.002, beta1 = 0.5, n_epoch=3000):
        
        self.generator=generator
        self.dynamics = dynamics
        
        self.lr=lr
        self.beta1=beta1
        self.n_epoch=3000
        self.optimizer = optim.Adam(self.generator.parameters(),
                                    lr=self.lr, betas=(self.beta1, 0.999))
        
        self.losses=[]
        self.initial_condition=0
        
    def J_obs(self, X, Rm1, Obs):
    
        # Quadratic observational error // Mahalanobis distance
        j = ((Obs-X)*Rm1*(Obs-X)).mean()
    
        return j

    def fit(self, noise, Obs, Rm1):
        
        for i in range(self.n_epoch):
            
            self.generator.zero_grad()
            loss=0
            ic_gen=self.generator((noise).unsqueeze(0)).squeeze(0)
        
            X=torch.zeros(Obs.shape)
            X[0,:,:,:]=ic_gen
        
            for t in range(Obs.shape[0]-1):
            
                X[t+1,:,:,:]=self.dynamics.forward(X[t,:,:,:].clone())
        
            loss=self.J_obs(X, Rm1, Obs)
            self.losses.append(loss.item())
        
            #keep the generated initial condition minimizing loss
            if loss.item()==min(self.losses):
                self.initial_condition=ic_gen.detach()
        
            loss.backward(retain_graph=True)
            self.optimizer.step()