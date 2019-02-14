import numpy as np
import numpy.random as npr
from tqdm import tqdm
from trslds.models import TroSLDS
from numpy import newaxis as na
from trslds import utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from trslds import initialize as init
from trslds import plotting
import copy
import seaborn as sns
from scipy.integrate import odeint

color_names = ["dirty yellow", "leaf green","red", "orange"]

colors_leaf = sns.xkcd_palette(color_names)
npr.seed(0)



def resample(no_samples, trslds):
    trslds._initialize_polya_gamma()  # Initialize polya-gamma rvs
    for m in tqdm(range(no_samples)):
        trslds._sample_emission()  # sample emission parameters
        trslds._sample_hyperplanes()  # sample hyperplanes
        trslds._sample_dynamics()  # Sample dynamics of tree
        trslds._sample_discrete_latent()  # Sample discrete latent states
        trslds._sample_continuous_latent()  # Sample continuous latent state
    return trslds

# In[]:

def noisy_FitzHugh(dt, T, start):
    v=np.zeros(T + 1)
    w=np.zeros(T + 1)
    v[0]=start[0]
    w[0]=start[1]
    
    for t in range(T):
        I=np.random.normal(0.7,0.2,1)*np.sqrt(dt/0.1)
        dv=(v[t]-np.power(v[t],3)/3-w[t]+I)*dt
        dw=0.08*dt*(v[t]+0.7-0.8*w[t])
        v[t+1]=v[t]+dv
        w[t+1]=w[t]+dw
    
    states=np.zeros((2,T + 1))
    states[0,:]=v
    states[1,:]=w
    return states

def FitzHugh(dt, T, start):
    v=np.zeros(T + 1)
    w=np.zeros(T + 1)
    v[0]=start[0]
    w[0]=start[1]
    
    for t in range(0,T):
        I = 0.7
        dv=(v[t]-np.power(v[t],3)/3-w[t]+I)*dt
        dw=0.08*dt*(v[t]+0.7-0.8*w[t])
        v[t+1]=v[t]+dv
        w[t+1]=w[t]+dw
    
    states=np.zeros((2,T + 1))
    states[0,:]=v
    states[1,:]=w
    return states


# In[]:
"Generate training data"
D_in = 2
D_out = 50
K = 4
no_realizations = 100
T = 400
dt = 0.1
starting_pts = 6*npr.rand(D_in, no_realizations) - 3
C = np.hstack((2*npr.rand(D_out, D_in) - 1, np.zeros((D_out,1))))
Xtrue = []
Y = []
Ysmooth = []
fig = plt.figure()
ax = fig.add_subplot(111)
for idx in range(D_out):
    xt = noisy_FitzHugh(dt, T, starting_pts[:, idx])
    #Demean 
    xt = xt - np.mean(xt, axis=1)[:, na]
    Xtrue.append(xt + 0)
    P = 1/(1 + np.exp(-(C[:, :-1] @ xt[:, 1:] + C[:, -1][:, na])))
    yt = npr.binomial( 1, P)
    Y.append(yt + 0)
    #Smooth using gaussian kernel smoother
    sigma = 25
    window = 400
    ysmooth = utils.gaussian_kernel_smoother(yt, sigma, window)
    Ysmooth.append(ysmooth + 0)
    ax.plot(ysmooth[0, :])


# In[]:
"Initalize values"
max_epochs = 1
batch_size = 128
lr = 0.0001
#Instead of passing in spikes, pass in smoothed spikes to do PCA on
A, C, R, X, Z, Path, possible_paths, leaf_path, leaf_nodes = init.initialize(Ysmooth, D_in, K, max_epochs, batch_size,
                                                                             lr)
Qstart = np.repeat(np.eye(D_in)[:, :, na], K, axis=2)
Sstart = np.eye(D_out)

kwargs = {'D_in': D_in, 'D_out': D_out, 'K': K, 'dynamics': A, 'dynamics_noise': Qstart, 'emission': C,
          'emission_noise': Sstart,
          'hyper_planes': R, 'possible_paths': possible_paths, 'leaf_path': leaf_path, 'leaf_nodes': leaf_nodes,
          'scale': 0.01}
trslds = TroSLDS(**kwargs) #Instantiiate the model


#Add data to model
for idx in range(len(Y)):
    trslds._add_data(X[idx], Y[idx], Z[idx], Path[idx])

fig = plt.figure()
ax = fig.add_subplot(111)
for idx in range(len(X)):
    ax.plot(X[idx][0, 1:], X[idx][1, 1:])
    
    