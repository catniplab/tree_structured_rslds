import numpy as np
import numpy.random as npr
from tqdm import tqdm
from trslds.models import TroSLDS
from numpy import newaxis as na
from trslds import utils
import matplotlib.pyplot as plt
from trslds import initialize as init
from trslds import plotting
import seaborn as sns
color_names = ["windows blue", "leaf green","red", "orange"]
colors_leaf = sns.xkcd_palette(color_names)
npr.seed(0)

D_in = 2 #Dimension of latent states
D_out = 2 #Dimension of observation space
K = 3 #Number of possible discrete latent states
# In[]:
def simulate_circle(no_realizations, Tmin=400, Tmax=800):
    # Create the tree
    K = 3
    depth, leaf_path, possible_paths, leaf_nodes = utils.create_balanced_binary_tree(K)

    # Define emission parameters
    C = np.zeros((D_out, D_in + 1))
    C[:, :-1] = np.eye(D_out)
    C[0, 1] = 2
    S = .001 * np.eye(D_out)

    # Create dynamics
    theta_1 = -.15 * np.pi / 2
    theta_2 = -.05 * np.pi / 2
    theta_3 = -.35*np.pi/2

    A = []
    A.append(np.zeros((D_in, D_in + 1, 1)))

    At = np.zeros((D_in, D_in + 1, 2))
    At[:, :-1, 0] = np.array([[np.cos(theta_3), -np.sin(theta_3)], [np.sin(theta_3), np.cos(theta_3)]])
    At[:, :, 1] *= np.nan
    A.append(At)

    At = np.zeros((D_in, D_in + 1, 4))
    At[:, :, 0] *= np.nan
    At[:, :, 1] *= np.nan
    At[:, :-1, 2] = np.array([[np.cos(theta_1), -np.sin(theta_1)], [np.sin(theta_1), np.cos(theta_1)]])
    At[:, :-1, 3] = np.array([[np.cos(theta_2), -np.sin(theta_2)], [np.sin(theta_2), np.cos(theta_2)]])
    A.append(At)

    Q = np.repeat(.001 * np.eye(D_in)[:, :, na], K, axis=2)  # Noise covariance

    # Create hyperplanes
    R_par = np.zeros((D_in + 1, 1))
    R_par[0, 0] = 100
    R_par[1, 0] = 100

    R = []
    R.append(R_par)
    R_temp = np.zeros((D_in + 1, 2))
    R_temp[:, 0] *= np.nan  # Left hyperplane
    R_temp[:-1, 1] = np.array([-100, 100])  # Right hyperplane
    R.append(R_temp)

    kwargs = {'D_in': D_in, 'D_out': D_out, 'K': K, 'dynamics': A, 'dynamics_noise': Q, 'emission': C,
              'emission_noise': S, 'hyper_planes': R, 'possible_paths': possible_paths, 'leaf_path':
                  leaf_path, 'leaf_nodes': leaf_nodes}

    true_model = TroSLDS(**kwargs)  # Create model

    # Generate data from model
    Xreal = []
    Yreal = []
    Zreal = []
    starting_pts = npr.uniform(-5, 5, (D_in, no_realizations))
    for reals in tqdm(range(no_realizations)):
        T = npr.randint(Tmin, Tmax + 1)
        y, x, z = true_model._generate_data(T, starting_pts[:, reals])
        Xreal.append(x)
        Yreal.append(y)
        Zreal.append(z)

    return Xreal, Yreal, Zreal, true_model

# In[]:
def resample(no_samples, trslds):
    trslds._initialize_polya_gamma()  # Initialize polya-gamma rvs
    for m in tqdm(range(no_samples)):
        trslds._sample_emission()  # sample emission parameters
        trslds._sample_hyperplanes()  # sample hyperplanes
        trslds._sample_dynamics()  # Sample dynamics of tree
        trslds._sample_discrete_latent()  # Sample discrete latent states
        trslds._sample_continuous_latent()  # Sample continuous latent states

    return trslds

def plot_rotated_vf(trslds, transform, xmin=-15, xmax=15, ymin=-15, ymax=15, delta=0.1):

    # Plot vector field and probability contour plot of inferred model projected onto true latent space
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    X, Y, arrows = plotting.rot_vector_field(trslds.Aleaf, trslds.R, xmin, xmax, ymin, ymax, delta, trslds.depth, trslds.leaf_paths,
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

# In[]:
#if __name__ == "__main__":
#First generate data from true model
no_realizations = 50
Xtrue, Y, Ztrue, true_model = simulate_circle(no_realizations)


# In[]:
#Plot trajectories
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
for idx in tqdm(range(no_realizations)):
    ax.scatter(Xtrue[idx][0, np.where(Ztrue[idx] == 0)], Xtrue[idx][1, np.where(Ztrue[idx] == 0)],
               color='green')
    ax.scatter(Xtrue[idx][0, np.where(Ztrue[idx] == 1)], Xtrue[idx][1, np.where(Ztrue[idx] == 1)],
               color='red')
    ax.scatter(Xtrue[idx][0, np.where(Ztrue[idx] == 2)], Xtrue[idx][1, np.where(Ztrue[idx] == 2)],
               color='blue')
ax.set_title('true latents')
fig.show()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# In[]
"Lets see if we can learn the model using TrSLDS. First, let's initialize the parameters."
batch_size = 256
max_epochs = 100
lr = 1e-4
A, C, R, X, Z, Path, possible_paths, leaf_path, leaf_nodes = init.initialize(Y, D_in, K, max_epochs, batch_size,
                                                                             lr)
fig = plt.figure()
ax = fig.add_subplot(111)
for idx in range(no_realizations):
    ax.scatter(X[idx][0, np.where(Z[idx][:-1] == 0)], X[idx][1, np.where(Z[idx][:-1] == 0)],
               color='green')
    ax.scatter(X[idx][0, np.where(Z[idx][:-1] == 1)], X[idx][1, np.where(Z[idx][:-1] == 1)],
               color='red')
    ax.scatter(X[idx][0, np.where(Z[idx][:-1] == 2)], X[idx][1, np.where(Z[idx][:-1] == 2)],
               color='blue')
ax.set_title('pca latents')
fig.show()

fig = plt.figure()
ax = fig.add_subplot(111)
for idx in range(no_realizations):
    ax.scatter(Y[idx][0, np.where(Ztrue[idx][:-1] == 0)], Y[idx][1, np.where(Ztrue[idx][:-1] == 0)],
               color='green')
    ax.scatter(Y[idx][0, np.where(Ztrue[idx][:-1] == 1)], Y[idx][1, np.where(Ztrue[idx][:-1] == 1)],
               color='red')
    ax.scatter(Y[idx][0, np.where(Ztrue[idx][:-1] == 2)], Y[idx][1, np.where(Ztrue[idx][:-1] == 2)],
               color='blue')
ax.set_title('Observations')
fig.show()
Qstart = np.repeat(np.eye(D_in)[:, :, na], K, axis=2)
Sstart = np.eye(D_out)

# In[]
kwargs = {'D_in': D_in, 'D_out': D_out, 'K': K, 'dynamics': A, 'dynamics_noise': Qstart, 'emission': C,
          'emission_noise': Sstart,
          'hyper_planes': R, 'possible_paths': possible_paths, 'leaf_path': leaf_path, 'leaf_nodes': leaf_nodes,
          'scale': 0.5}
trslds = TroSLDS(**kwargs)  # Instantiiate the model

# Add data to model
for idx in range(len(Y)):
    trslds._add_data(X[idx], Y[idx], Z[idx], Path[idx])

# Perform Gibbs to train the model
no_samples = 200
trslds = resample(no_samples, trslds)

# Obtain transformation matrix from inferred latent space to true latent space
transform = utils.projection(Xtrue, trslds.x)
Xinferr = trslds.x
# Project inferred latent space to true latent space
Xinferr = [transform[:, :-1] @ Xinferr[idx] + transform[:, -1][:, na] for idx in range(len(Xinferr))]
Zinferr = trslds.z

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
for idx in tqdm(range(no_realizations)):
    ax.scatter(Xinferr[idx][0, np.where(Zinferr[idx] == 0)], Xinferr[idx][1, np.where(Zinferr[idx] == 0)],
               color='green')
    ax.scatter(Xinferr[idx][0, np.where(Zinferr[idx] == 1)], Xinferr[idx][1, np.where(Zinferr[idx] == 1)],
               color='red')
    ax.scatter(Xinferr[idx][0, np.where(Zinferr[idx] == 2)], Xinferr[idx][1, np.where(Zinferr[idx] == 2)],
               color='blue')
    ax.scatter(Xinferr[idx][0, np.where(Zinferr[idx] == 3)], Xinferr[idx][1, np.where(Zinferr[idx] == 3)],
               color='purple')
fig.show()
# Plot rotated vector field colored by probability of latent discrete state assignment
plot_rotated_vf(trslds, transform, xmin=-7, xmax=7, ymin=-7, ymax=7, delta=0.1)

transform_t = np.hstack((np.eye(2), np.zeros((2, 1))))
plot_rotated_vf(true_model, transform_t, xmin=-7, xmax=7, ymin=-7, ymax=7, delta=0.1)
