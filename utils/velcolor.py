## Middlebury color representation
##
## according to Matlab source code of Deqing Sun (dqsun@cs.brown.edu)
## Python translation by Dominique Bereziat (dominique.bereziat@lip6.fr)
import matplotlib.pyplot as plt
import numpy as np

def colorWheel():
    """ internal function """
    RY,YG,GC = 15,6,4
    CB,BM,MR = 11,13,6

    ncols = RY+YG+GC+CB+BM+MR
    colwheel = np.zeros((ncols,3),dtype=np.uint8)

    col = 0
    colwheel[:RY,0] = 255
    colwheel[:RY,1] = np.floor(255*np.arange(RY)/RY)
    col += RY

    colwheel[col:col+YG,0] = 255 - np.floor(255*np.arange(YG)/YG)
    colwheel[col:col+YG,1] = 255
    col += YG

    colwheel[col:col+GC,1] = 255
    colwheel[col:col+GC,2] = np.floor(255*np.arange(GC)/GC)
    col += GC

    colwheel[col:col+CB,1] = 255 - np.floor(255*np.arange(CB)/CB)
    colwheel[col:col+CB,2] = 255
    col += CB

    colwheel[col:col+BM,2] = 255
    colwheel[col:col+BM,0] = np.floor(255*np.arange(BM)/BM)
    col += BM

    colwheel[col:col+MR,2] = 255 - np.floor(255*np.arange(MR)/MR)
    colwheel[col:col+MR,0] = 255

    return colwheel

def computeColor(U1,V1,norma=False):
    """ bla bla """
    
    U = U1.copy() # changement fait ici car computeColor modifiait son entrée
    V = V1.copy()

    nanIdx = np.isnan(U) | np.isnan(V)
    U[nanIdx] = 0
    V[nanIdx] = 0

    colwheel = colorWheel()
    ncols = colwheel.shape[0]

    rad = np.sqrt(U**2+V**2)
    if norma:
        U /= 2*np.max(rad)
        V /= 2*np.max(rad)
        rad = np.sqrt(U**2+V**2)
    
    a = np.arctan2(-V,-U)/np.pi

    fk = (a+1)/2 * (ncols-1) # -1~1 maped to 0~ncols-1
    k0 = np.uint8(np.floor(fk))
    k1 = (k0+1)%ncols
    f = fk - k0
    img = np.empty((U.shape[0],U.shape[1],3),dtype=np.uint8)

    for i in range(3):
        tmp = colwheel[:,i]
        col0 = tmp[k0]/255;
        col1 = tmp[k1]/255;
        col = (1-f)*col0 + f*col1
        idx = rad <=1
        col[idx] = 1 - rad[idx] * (1-col[idx])
        col[~idx] *= 0.75
        img[:,:,2-i] = np.uint8(np.floor(255*col*(1-nanIdx)))
    return img


def plot_w(U,V,title = 'W',
           normalization=True,quiver=False,q_alpha = 0.5,q_scale =0.1,sub=4):
    
    x,y = np.meshgrid(np.arange(0,U.shape[0],sub),np.arange(0,U.shape[1],sub))

    plt.imshow(computeColor(U.numpy(),V.numpy(),norma = normalization))#,origin='lower')
    #plt.axis('off')
    
    if quiver:
        
        plt.quiver(y,x,U[x,y],-V[x,y],alpha = q_alpha,scale = q_scale)
        
    plt.title(title)
    plt.yticks([])
    plt.xticks([])
