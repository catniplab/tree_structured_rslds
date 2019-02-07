import numpy as np
import numpy.random as npr
from tqdm import tqdm
from models import TroSLDS
from numpy import newaxis as na
import utils
import matplotlib.pyplot as plt
import initialize as init
npr.seed(0)
# In[1]:
D_in = 2
D_out = 2
# Create the tree
K = 4
depth, leaf_path, possible_paths, leaf_nodes = utils.create_balanced_binary_tree(K)

#Define emission parameters
C = np.zeros((D_out, D_in + 1))
C[:, :-1] = np.eye(D_out)
C[0, 1] = 2
S = .1 * np.eye(D_out)

#Create dynamics
A = []
A.append(np.zeros((D_in, D_in + 1, 1)))
A.append(np.zeros((D_in, D_in + 1, 2)))
At = np.zeros((D_in, D_in + 1, 4))
theta_1 = -.15*np.pi/2
theta_2 = -.05*np.pi/2

At[:, :-1, 0] = np.eye(D_in)
At[:, -1, 0] = np.array([.25, 0])

At[:, :-1, 1] = np.array([[np.cos(theta_1), -np.sin(theta_1)], [np.sin(theta_1), np.cos(theta_1)]])
At[:, -1, 1] = ((-At[:, :-1, 1] + np.eye(D_in)) @ np.array([4, 0])[:, na]).flatten()

At[:, :-1, 2] = np.array([[np.cos(theta_2), -np.sin(theta_2)], [np.sin(theta_2), np.cos(theta_2)]])
At[:, -1, 2] = ((-At[:, :-1, 2] + np.eye(D_in)) @ np.array([-4, 0])[:, na]).flatten()


At[:, :-1, 3] = np.eye(D_in)
At[:, -1, 3] = np.array([-.05, 0])

A.append(At)

Q = np.repeat(.001*np.eye(D_in)[:, :, na], K, axis=2) #Noise covariance

#Create hyperplanes
R_par = np.zeros((D_in + 1, 1))
R_par[0, 0] = 100
R_par[1, 0] = 100
r_par = np.array([0.0])

R = []
R.append(R_par)
R_temp = np.zeros((D_in + 1,2))
R_temp[:-1, 0] = np.array([-100, 100]) #Left hyperplane
R_temp[:-1, 1] = np.array([-100, 100]) #Right hyperplane
R.append(R_temp)

kwargs = {'D_in': D_in, 'D_out': D_out, 'K': K, 'dynamics': A, 'dynamics_noise': Q, 'emission': C, 'emission_noise': S,
          'hyper_planes': R, 'possible_paths': possible_paths, 'leaf_path': leaf_path, 'leaf_nodes': leaf_nodes}

true_model = TroSLDS(**kwargs) #Create model

# In[1]:
#Generate data from model
no_realizations = 50
Tmin = 400
Tmax = 800
Xreal = []
Yreal = []
Zreal = []
starting_pts = npr.uniform(-10, 10, (D_in, no_realizations))
for reals in tqdm(range(no_realizations)):
    T = npr.randint(Tmin, Tmax + 1)
    y, x, z = true_model._generate_data(T, starting_pts[:, reals])
    Xreal.append(x)
    Yreal.append(y)
    Zreal.append(z)


# In[2]:
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

for idx in tqdm(range(no_realizations)):
    ax.scatter(Xreal[idx][0, np.where(Zreal[idx] == 0)], Xreal[idx][1, np.where(Zreal[idx] == 0)], color='green')
    ax.scatter(Xreal[idx][0, np.where(Zreal[idx] == 1)], Xreal[idx][1, np.where(Zreal[idx] == 1)], color='red')
    ax.scatter(Xreal[idx][0, np.where(Zreal[idx] == 2)], Xreal[idx][1, np.where(Zreal[idx] == 2)], color='blue')
    ax.scatter(Xreal[idx][0, np.where(Zreal[idx] == 3)], Xreal[idx][1, np.where(Zreal[idx] == 3)], color='purple')

fig.show()

# In[]:
"Lets see if we can learn the model using TrSLDS. First, let's initialize the parameters."
batch_size = 256
max_epochs = 100
lr = 1e-3
A, C, R, X, Z, Path, possible_paths, leaf_path, leaf_nodes = init.initialize(Yreal, D_in, K, max_epochs, batch_size, lr)
Qstart = np.repeat(np.eye(D_in)[:, :, na], K, axis=2)
Sstart = np.eye(D_out)


# In[]:
kwargs = {'D_in': D_in, 'D_out': D_out, 'K': K, 'dynamics': A, 'dynamics_noise': Qstart, 'emission': C, 'emission_noise': Sstart,
          'hyper_planes': R, 'possible_paths': possible_paths, 'leaf_path': leaf_path, 'leaf_nodes': leaf_nodes, 'scale':0.5}
trslds = TroSLDS(**kwargs)
"Perform Gibbs sampling to learn the parameters"
for idx in range(no_realizations):
    trslds._add_data(X[idx], Yreal[idx], Z[idx], Path[idx])

no_samples = 10
trslds._initialize_polya_gamma() # Initialze polya-gamma rvs
for m in tqdm(range(no_samples)):
    trslds._sample_emission() #sample emission parameters
    trslds._sample_hyperplanes() #sample hyperplanes
    trslds._sample_dynamics() #Sample dynamics of tree
    trslds._sample_discrete_latent() #Sample discrete latent states
#    trslds._sample_pg()
    trslds._sample_continuous_latent() #Sample continuous latent states


# In[]:
"Plot inferred latent states"
Xinferr = trslds.x
#Project onto real space
Xreals = np.hstack(Xreal).T
Xrot = np.hstack(Xinferr).T
Xrot = np.hstack((Xrot, np.ones((Xrot[:, 0].size, 1))))
transform = np.linalg.lstsq(Xrot, Xreals)[0].T
Xinferr = [transform[:, :-1] @ Xinferr[idx] + transform[:, -1][:, na] for idx in range(len(Xinferr))]

Zinferr = trslds.z
fig = plt.figure()
ax = fig.add_subplot(111)
for idx in tqdm(range(no_realizations)):
    ax.scatter(Xinferr[idx][0, np.where(Zinferr[idx] == 0)], Xinferr[idx][1, np.where(Zinferr[idx] == 0)], color='green')
    ax.scatter(Xinferr[idx][0, np.where(Zinferr[idx] == 1)], Xinferr[idx][1, np.where(Zinferr[idx] == 1)], color='red')
    ax.scatter(Xinferr[idx][0, np.where(Zinferr[idx] == 2)], Xinferr[idx][1, np.where(Zinferr[idx] == 2)], color='blue')
    ax.scatter(Xinferr[idx][0, np.where(Zinferr[idx] == 3)], Xinferr[idx][1, np.where(Zinferr[idx] == 3)], color='purple')
    



