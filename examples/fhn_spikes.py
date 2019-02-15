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
D_out = 100
K = 4
no_realizations = 100
T = 400
dt = 0.1
starting_pts = np.zeros((2, no_realizations) )
starting_pts[0,:] = 4*npr.rand(no_realizations)-2
starting_pts[1,:] = 2.5*npr.rand(no_realizations)-0.5
C = np.hstack((2*npr.rand(D_out, D_in) - 1, np.zeros((D_out,1))))
Xtrue = []
Y = []
Ysmooth = []
fig = plt.figure()
ax = fig.add_subplot(111)
for idx in range(D_out):
    xt = noisy_FitzHugh(dt, T, starting_pts[:, idx])
    #Demean 
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
max_epochs = 200
batch_size = 128
lr = 0.0001
#Instead of passing in spikes, pass in smoothed spikes to do PCA on
A, C, R, X, Z, Path, possible_paths, leaf_path, leaf_nodes = init.initialize(Ysmooth, D_in, K, max_epochs, batch_size,
                                                                             lr)
Qstart = np.repeat(np.eye(D_in)[:, :, na], K, axis=2)
Sstart = np.eye(D_out)

#Have to pass in bern=True for spike trains
kwargs = {'D_in': D_in, 'D_out': D_out, 'K': K, 'dynamics': A, 'dynamics_noise': Qstart, 'emission': C,
          'emission_noise': Sstart,
          'hyper_planes': R, 'possible_paths': possible_paths, 'leaf_path': leaf_path, 'leaf_nodes': leaf_nodes,
          'scale': 0.01, 'bern':True}
trslds = TroSLDS(**kwargs) #Instantiiate the model


#Add data to model
for idx in range(len(Y)):
    trslds._add_data(X[idx], Y[idx], Z[idx], Path[idx])

#Visualize initalization of latent states to see if it looks okay
fig = plt.figure()
ax = fig.add_subplot(111)
for idx in range(len(X)):
    ax.plot(X[idx][0, 1:], X[idx][1, 1:])
    
    
# In[]:
no_samples = 500 #For ICLR we ran for 1,000 samples but it converges rather quickly. 100 should be fine.
trslds = resample(no_samples, trslds)

# In[]:
# Obtain transformation matrix from inferred latent space to true latent space
transform = utils.projection(Xtrue, trslds.x)
Xinferr = trslds.x
# Project inferred latent space to true latent space
Xinferr = [transform[:, :-1] @ Xinferr[idx] + transform[:, -1][:, na] for idx in range(len(Xinferr))]
Zinferr = trslds.z

# In[]:
At, Qt = utils.MAP_dynamics(trslds.x, trslds.u, trslds.z, trslds.A, trslds.Q, trslds.nux, trslds.lambdax, 
                      trslds.Mx, trslds.Vx, trslds.scale, trslds.leaf_nodes, K, trslds.depth, 10000)

# In[]:
def fhn_vf(xmin, xmax, ymin, ymax, delta):
    x, y = np.arange(xmin,xmax, delta), np.arange(ymin,ymax, delta)
    X,Y = np.meshgrid(x,y)
    points = np.zeros( (X[:,0].size, X[0,:].size, 2) )
    points[:,:,0] = X
    points[:,:, 1] = Y
    arrows = np.zeros( points.shape )
    for row in range(X[:, 0].size):
        for col in range(X[0, :].size):
            I = 0.7
            dv=(X[row, col]-np.power(X[row, col],3)/3-Y[row, col]+I)
            dw=0.08*(X[row, col]+0.7-0.8*Y[row, col])
            arrows[row, col, 0] = dv
            arrows[row, col, 1] = dw
    return X, Y, arrows

# In[]:
fig = plt.figure(figsize=(20, 20))
gs = gridspec.GridSpec(2, 2)

#Plot true latents
ax = fig.add_subplot(gs[0, 0])
for idx in range(len(Xtrue)):
    ax.plot(Xtrue[idx][0, :], Xtrue[idx][1, :])
ax.set_title('true latent states')
#Plot inferred latents
ax = fig.add_subplot(gs[1, 0])
for idx in range(len(Xtrue)):
    ax.plot(Xinferr[idx][0, :], Xinferr[idx][1, :])
ax.set_title('inferred latent states')    


#Plot true vector field
xmin = -2
xmax = 2
ymin = -1
ymax = 2
delta = 0.1

ax = fig.add_subplot(gs[0, 1])
X, Y, arrows = fhn_vf(xmin, xmax, ymin, ymax, delta)
norm = np.sqrt(arrows[:, :, 0] ** 2 + arrows[:, :, 1] ** 2)
U = arrows[:, :, 0] / norm
V = arrows[:, :, 1] / norm

ax.streamplot(X, Y, U, V, color=np.log(norm), cmap='plasma_r')
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

#Plot learned vector field

ax = fig.add_subplot(gs[1, 1])
X, Y, arrows = plotting.rot_vector_field(At[-1], trslds.R, xmin, xmax, ymin, ymax, delta, trslds.depth, trslds.leaf_paths,
                                             trslds.K, transform)
norm = np.sqrt(arrows[:, :, 0] ** 2 + arrows[:, :, 1] ** 2)
U = arrows[:, :, 0] / norm
V = arrows[:, :, 1] / norm

ax.streamplot(X, Y, U, V, color=np.log(norm), cmap='plasma_r')

X, Y, color = plotting.rot_contour_plt(trslds.R, xmin, xmax, ymin, ymax, delta, trslds.depth, trslds.leaf_paths,
                                       trslds.K, transform)

for k in range(K):
    start = np.array([1., 1., 1., 0.])
    end = np.concatenate((colors_leaf[k], [0.5]))
    cmap = plotting.gradient_cmap([start, end])
    im1 = ax.imshow(color[:, :, k],
                    extent=[xmin, xmax, ymin, ymax],
                    vmin=0, vmax=1, cmap=cmap, origin='lower')
    ax.set_aspect('auto')
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])




fig.show()
fig.tight_layout()
    