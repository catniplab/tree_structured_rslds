import numpy as np
import numpy.random as npr
from tqdm import tqdm
from models import TroSLDS
from numpy import newaxis as na
import utils
import matplotlib.pyplot as plt
import initialize as init
import plotting
import seaborn as sns
color_names = ["windows blue", "leaf green","red", "orange"]
colors_leaf = sns.xkcd_palette(color_names)
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

# In[2]:
"Check to see if plotting code works"
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
xmin = -15
xmax = 15
ymin = -15
ymax = 15
delta = 0.1


X, Y, arrows = plotting.vector_field(true_model.Aleaf, true_model.R, xmin, xmax, ymin, ymax, delta, depth, leaf_path, K)
norm = np.sqrt(arrows[:, :, 0] ** 2 + arrows[:, :, 1] ** 2)
U = arrows[:, :, 0]/norm
V = arrows[:, :, 1]/norm

ax.streamplot(X, Y, U, V, color=np.log(norm), cmap='plasma_r')

# In[3]:
"Check to see if contour plot works"
X, Y, color = plotting.contour_plt(true_model.R, xmin, xmax, ymin, ymax, delta, depth, leaf_path, K)

for k in range(K):
    start = np.array([1., 1., 1., 0.])
    end = np.concatenate((colors_leaf[k ], [0.5]))
    cmap = plotting.gradient_cmap([start, end])
    im1 = ax.imshow(color[:,:,k],
                    extent=[xmin, xmax, ymin,ymax],
                    vmin=0, vmax=1, cmap=cmap, origin='lower')
    ax.set_aspect('auto')
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

fig.show()