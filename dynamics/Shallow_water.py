import torch 
import numpy as np
import torch.nn.functional as func

class SW():
    
    def __init__(self, dx, dy, dt_factor=0.5,
                 H=100, g=9.81, unsqz=False):
        
        self.H = H
        self.g = g
        
        self.dx = dx
        self.dy = dy
        self.dt = dt_factor*min(dx, dy)/np.sqrt(g*H)
        
        self.unsqz = unsqz

    def forward(self, X):
        
        X_int=torch.zeros(X.shape)
        
        #state variable
        eta=X[0,:,:]
        u=X[1,:,:]
        v=X[2,:,:]
        
        eta_int=torch.zeros(eta.shape)
        u_int=torch.zeros(u.shape)
        v_int=torch.zeros(v.shape)
        
        #path variable
        h_e = torch.zeros(u.shape)
        h_w = torch.zeros(u.shape)
        h_n = torch.zeros(u.shape)
        h_s = torch.zeros(u.shape)
        uhwe = torch.zeros(u.shape)
        vhns = torch.zeros(u.shape)
        
        H=self.H
        g=self.g
        dt=self.dt
        dx=self.dx
        dy=self.dy
        
        # ------------ u,v integration --------------
        u_int[:-1, :] = u[:-1, :] - g*dt/dx*(eta[1:, :] - eta[:-1, :])
        v_int[:, :-1] = v[:, :-1] - g*dt/dy*(eta[:, 1:] - eta[:, :-1])

        # ------------ eta integration --------------
        h_e[:-1, :] = torch.where(u_int[:-1, :] > 0, eta[:-1, :] + H, eta[1:, :] + H)
        h_e[-1, :] = eta[-1, :] + H

        h_w[0, :] = eta[0, :] + H
        h_w[1:, :] = torch.where(u_int[:-1, :] > 0, eta[:-1, :] + H, eta[1:, :] + H)

        h_n[:, :-1] = torch.where(v_int[:, :-1] > 0, eta[:, :-1] + H, eta[:, 1:] + H)
        h_n[:, -1] = eta[:, -1] + H

        h_s[:, 0] = eta[:, 0] + H
        h_s[:, 1:] = torch.where(v_int[:, :-1] > 0, eta[:, :-1] + H, eta[:, 1:] + H)

        uhwe[0, :] = u_int[0, :]*h_e[0, :]
        uhwe[1:, :] = u_int[1:, :]*h_e[1:, :] - u_int[:-1, :]*h_w[1:, :]

        vhns[:, 0] = v_int[:, 0]*h_n[:, 0]
        vhns[:, 1:] = v_int[:, 1:]*h_n[:, 1:] - v_int[:, :-1]*h_s[:, 1:]
        
        eta_int[:, :] = eta[:, :] - dt*(uhwe[:, :]/dx + vhns[:, :]/dy)

        # ----------------- X integration -------------------
        X_int[0,:,:]=eta_int
        X_int[1,:,:]=u_int
        X_int[2,:,:]=v_int

        return X_int