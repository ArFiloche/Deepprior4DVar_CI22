import torch
import torch.optim as optim
from scipy import interpolate

#from .cost_function import*

class strong_4DVar():
    
    
    def __init__(self, dynamics,regul=None,
                 optimizer=optim.LBFGS(
                     [torch.zeros(0)],
                     lr=0.75, 
                     max_iter=500)
                ):
                                         #tolerance_grad=0,
                                         #tolerance_change=0)
        
        self.dynamics = dynamics
        
        #self.alpha=alpha
        #self.beta=beta 
        self.regul=regul
        
        self.optimizer = optimizer
        self.n_iter = 0
        self.losses=[]
        self.convergence=1
    
    def Forward(self, X0):
        
        # Initialize state
        X = torch.zeros(self.DAw)
        X[0,:,:,:] = X0
        
        # Time integration
        for t in range(1,self.T):
            X[t,:,:,:] = self.dynamics.forward(X[(t-1),:,:,:].clone())
            
        return X
        
    def J_obs(self, X, Rm1, Obs):
    
        # Quadratic observational error
        j = ((Obs-X)*Rm1*(Obs-X)).mean()
        
        # vanish
        #j = j*(((Rm1==Rm1).sum().item()/Rm1!=0).sum().item())
    
        return j
    
   # def J_reg(self, X0):
        
        # velocity field smoothness regularization
        
        #return j

    def fit(self, Obs, Rm1):
        
        # Obs & Covariance
        self.DAw = Obs.shape
        self.T = self.DAw[0]
        self.Obs = Obs
        self.Rm1 = Rm1
        
        # Background // first obs by default
        X_b = torch.Tensor(Obs[0,:,:,:])
        
        # eps_b, control paramaters
        self.eps_b = torch.zeros(X_b.shape)
        self.eps_b.requires_grad = True
        self.optimizer.param_groups[0]['params'][0] = self.eps_b
        
        def closure():
            
            self.optimizer.zero_grad()
            X0 = X_b + self.eps_b
            X = self.Forward(X0)
            
            if self.regul == None:
                loss  = self.J_obs(X,self.Rm1,self.Obs)
            else:
                loss  = self.J_obs(X,self.Rm1,self.Obs)+self.regul.J(X)

            # check for NaN
            if torch.isnan(loss).item() != 0:          
                print('Nan loss: failed to converge')
                self.convergence=0
                loss = torch.zeros(1,requires_grad = True)
            
            loss.backward(retain_graph=True)
            
            self.n_iter = self.n_iter + 1
            self.losses.append(loss)
            
            return loss
        
        loss = self.optimizer.step(closure)
        
        # Full state
        self.X = self.Forward(X_b + self.eps_b)
        
        # Initial condition
        self.initial_condition = self.X[0,:,:,:].detach()
        
        
#    def forecast(self, n_step=10):
#
#        X_forecast = self.X[-1,:,:,:].detach()
#
#        for i in range (0, n_step):
#
#            X_forecast = self.model.forward(X_forecast)
#            
#        return X_forecast