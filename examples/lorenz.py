import numpy as np
import numpy.random as npr
from tqdm import tqdm
from trslds.models import TroSLDS
from numpy import newaxis as na
from trslds import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from trslds import initialize as init
from trslds import plotting
import seaborn as sns
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
# Load in dataset
iclr_lorenz = np.load('iclr_lorenz.npy', allow_pickle=True)[()]
#Extract out the observations and latent states
Xtrue = iclr_lorenz['X']
Y = iclr_lorenz['Y']

D_out = Y[0][:, 0].size #Obtain dimension of observation space
D_in = 3  #Dimension of latent space
K = 4 #Number of discrete latent states

# In[]:
# Initialize the model
max_epochs = 200
batch_size = 128
lr = 0.0001
A, C, R, X, Z, Path, possible_paths, leaf_path, leaf_nodes = init.initialize(Y, D_in, K, max_epochs, batch_size,
                                                                             lr)
Qstart = np.repeat(np.eye(D_in)[:, :, na], K, axis=2)
Sstart = np.eye(D_out)

kwargs = {'D_in': D_in, 'D_out': D_out, 'K': K, 'dynamics': A, 'dynamics_noise': Qstart, 'emission': C,
          'emission_noise': Sstart,
          'hyper_planes': R, 'possible_paths': possible_paths, 'leaf_path': leaf_path, 'leaf_nodes': leaf_nodes,
          'scale': 0.8}
trslds = TroSLDS(**kwargs) #Instantiiate the model


#Add data to model
for idx in range(len(Y)):
    trslds._add_data(X[idx], Y[idx], Z[idx], Path[idx])


#Perform Gibbs to train the model
no_samples = 500
trslds = resample(no_samples, trslds)

# In[]:

# Obtain transformation matrix from inferred latent space to true latent space
transform = utils.projection(Xtrue, trslds.x)
Xinferr = trslds.x
# Project inferred latent space to true latent space
Xinferr = [transform[:, :-1] @ Xinferr[idx] + transform[:, -1][:, na] for idx in range(len(Xinferr))]
Zinferr = trslds.z

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for idx in tqdm(range(len(Y))):
    ax.scatter(Xinferr[idx][0, np.where(Zinferr[idx] == 0)], Xinferr[idx][1, np.where(Zinferr[idx] == 0)],
               Xinferr[idx][2, np.where(Zinferr[idx] == 0)], color='green')

    ax.scatter(Xinferr[idx][0, np.where(Zinferr[idx] == 1)], Xinferr[idx][1, np.where(Zinferr[idx] == 1)],
               Xinferr[idx][2, np.where(Zinferr[idx] == 1)], color='red')

    ax.scatter(Xinferr[idx][0, np.where(Zinferr[idx] == 2)], Xinferr[idx][1, np.where(Zinferr[idx] == 2)],
               Xinferr[idx][2, np.where(Zinferr[idx] == 2)], color='blue')

    ax.scatter(Xinferr[idx][0, np.where(Zinferr[idx] == 3)], Xinferr[idx][1, np.where(Zinferr[idx] == 3)],
               Xinferr[idx][2, np.where(Zinferr[idx] == 3)], color='purple')
fig.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for idx in range(len(Xtrue)):
    ax.plot(Xtrue[idx][0, :], Xtrue[idx][1, :], Xtrue[idx][2, :])

fig.show()


# In[]:
import matplotlib.gridspec as gridspec
#Make ICLR figure
fig = plt.figure()
gs = gridspec.GridSpec(2, 2)

"Real trajectories"
ax1 = fig.add_subplot(gs[0, 0], projection='3d')

for idx in range(len(Xtrue)):
    ax1.plot(Xtrue[idx][0, :], Xtrue[idx][1, :], Xtrue[idx][2, :])
    ax1.scatter(Xtrue[idx][0, 0], Xtrue[idx][1, 0], Xtrue[idx][2, 0], marker='x', color='red', s=40)
ax1.set_title('true latent trajectories')
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.set_zticklabels([])
xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
zlim = ax1.get_zlim()
ax1.set_xlabel('$x_1$', labelpad= 0, fontsize = 16)
ax1.set_ylabel('$x_2$', labelpad= .5, fontsize = 16)
ax1.set_zlabel('$x_3$', labelpad= 0, horizontalalignment='center', fontsize = 16)

"Plot inferred trajectories colored by inferred discrete latent states"
ax = fig.add_subplot(gs[1, 0], projection='3d')
for idx in tqdm(range(len(Xinferr))):
    for t in range(X[idx][0, :].size):
        ax.plot(Xinferr[idx][0, t:t+2], Xinferr[idx][1, t:t+2], Xinferr[idx][2, t:t+2], color=colors_leaf[int(Z[idx][t])])
fig.show()



"Plot generated trajectories from leaf node"
_, xnew, znew = trslds._generate_data(5000, X[2][:, 2], )
xnew = transform[:, :-1] @ xnew + transform[:, -1][:, na]
ax = fig.add_subplot(gs[0, 1], projection='3d')
ax.cla()
for t in range(xnew[0, :].size):
    ax.plot(xnew[0, t:t+2], xnew[1, t:t+2], xnew[2, t:t+2], color=colors_leaf[int(znew[t])])
fig.show()


