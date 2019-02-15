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
import seaborn as sns
color_names = ["windows blue", "leaf green","red", "orange"]
colors_leaf = sns.xkcd_palette(color_names)
npr.seed(0)

D_in = 2 #Dimension of latent states
D_out = 2 #Dimension of observation space
K = 4 #Number of possible discrete latent states
# In[]:
def simulate_tree_nascar(no_realizations, Tmin=400, Tmax=800):
    # Create the tree
    K = 4
    depth, leaf_path, possible_paths, leaf_nodes = utils.create_balanced_binary_tree(K)

    # Define emission parameters
    C = np.zeros((D_out, D_in + 1))
    C[:, :-1] = np.eye(D_out)
    C[0, 1] = 2
    S = .001 * np.eye(D_out)

    # Create dynamics
    A = []
    A.append(np.zeros((D_in, D_in + 1, 1)))
    A.append(np.zeros((D_in, D_in + 1, 2)))
    At = np.zeros((D_in, D_in + 1, 4))
    theta_1 = -.15 * np.pi / 2
    theta_2 = -.05 * np.pi / 2

    At[:, :-1, 0] = np.eye(D_in)
    At[:, -1, 0] = np.array([.25, 0])

    At[:, :-1, 1] = np.array([[np.cos(theta_1), -np.sin(theta_1)], [np.sin(theta_1), np.cos(theta_1)]])
    At[:, -1, 1] = ((-At[:, :-1, 1] + np.eye(D_in)) @ np.array([4, 0])[:, na]).flatten()

    At[:, :-1, 2] = np.array([[np.cos(theta_2), -np.sin(theta_2)], [np.sin(theta_2), np.cos(theta_2)]])
    At[:, -1, 2] = ((-At[:, :-1, 2] + np.eye(D_in)) @ np.array([-4, 0])[:, na]).flatten()

    At[:, :-1, 3] = np.eye(D_in)
    At[:, -1, 3] = np.array([-.05, 0])

    A.append(At)

    Q = np.repeat(.001 * np.eye(D_in)[:, :, na], K, axis=2)  # Noise covariance

    # Create hyperplanes
    R_par = np.zeros((D_in + 1, 1))
    R_par[0, 0] = 100
    R_par[1, 0] = 100

    R = []
    R.append(R_par)
    R_temp = np.zeros((D_in + 1, 2))
    R_temp[:-1, 0] = np.array([-100, 100])  # Left hyperplane
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
    starting_pts = npr.uniform(-10, 10, (D_in, no_realizations))
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

# In[]:
if __name__ == "__main__":
    #First generate data from true model
    no_realizations = 50
    Xtrue, Y, Ztrue, true_model = simulate_tree_nascar(no_realizations)


    "Lets see if we can learn the model using TrSLDS. First, let's initialize the parameters."
    batch_size = 256
    max_epochs = 100
    lr = 1e-3
    A, C, R, X, Z, Path, possible_paths, leaf_path, leaf_nodes = init.initialize(Y, D_in, K, max_epochs, batch_size,
                                                                                 lr)
    Qstart = np.repeat(np.eye(D_in)[:, :, na], K, axis=2)
    Sstart = np.eye(D_out)

    kwargs = {'D_in': D_in, 'D_out': D_out, 'K': K, 'dynamics': A, 'dynamics_noise': Qstart, 'emission': C,
              'emission_noise': Sstart,
              'hyper_planes': R, 'possible_paths': possible_paths, 'leaf_path': leaf_path, 'leaf_nodes': leaf_nodes,
              'scale': 0.5}
    trslds = TroSLDS(**kwargs) #Instantiiate the model

    #Add data to model
    for idx in range(len(Y)):
        trslds._add_data(X[idx], Y[idx], Z[idx], Path[idx])


    #Perform Gibbs to train the model
    no_samples = 100
    trslds = resample(no_samples, trslds)

   #Obtain transformation matrix from inferred latent space to true latent space
    transform = utils.projection(Xtrue, trslds.x)
    Xinferr = trslds.x
    #Project inferred latent space to true latent space
    Xinferr = [transform[:, :-1] @ Xinferr[idx] + transform[:, -1][:, na] for idx in range(len(Xinferr))]
    Zinferr = trslds.z

# In[]:
    "Get the mode of the conditional posterior over the dynamics"
    At, Qt = utils.MAP_dynamics(trslds.x, trslds.u, trslds.z, trslds.A, trslds.Q, trslds.nux, trslds.lambdax,
                                trslds.Mx, trslds.Vx, trslds.scale, trslds.leaf_nodes, K, trslds.depth, 10000)

# In[]:
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 3)

    # Plot true latent states colored by true discrete state
    ax = fig.add_subplot(gs[0, 0])
    for idx in tqdm(range(len(Xtrue))):
        for t in range(Xtrue[idx][0, :].size):
            ax.plot(Xtrue[idx][0, t:t + 2], Xtrue[idx][1, t:t + 2], color=colors_leaf[int(Ztrue[idx][t])])
    ax.set_title('true latent states')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

# In[]
    # Plot inferred latent states colored by inferred discrete state
    ax = fig.add_subplot(gs[1, 0])
    for idx in tqdm(range(len(Xtrue))):
        for t in range(Xtrue[idx][0, :].size):
            ax.plot(Xinferr[idx][0, t:t + 2], Xinferr[idx][1, t:t + 2], color=colors_leaf[int(Zinferr[idx][t])])
    ax.set_title('inferred latent states')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# In[]
    #Plot true vector field and color by discrete state
    ax = fig.add_subplot(gs[0, 1])
    xmin = xlim[0]
    xmax = xlim[1]
    ymin = ylim[0]
    ymax = ylim[1]
    delta = 1
    X, Y, arrows = plotting.vector_field(true_model.Aleaf, true_model.R, xmin, xmax, ymin, ymax, delta, true_model.depth,
                                         true_model.leaf_paths, true_model.K)
    norm = np.sqrt(arrows[:, :, 0] ** 2 + arrows[:, :, 1] ** 2)
    U = arrows[:, :, 0] / norm
    V = arrows[:, :, 1] / norm
    ax.streamplot(X, Y, U, V, color=np.log(norm), cmap='plasma_r')

    X, Y, color = plotting.contour_plt(true_model.R, xmin, xmax, ymin, ymax, delta, true_model.depth,
                                       true_model.leaf_paths, true_model.K)

    for k in range(K):
        start = np.array([1., 1., 1., 0.])
        end = np.concatenate((colors_leaf[k], [0.5]))
        cmap = plotting.gradient_cmap([start, end])
        im1 = ax.imshow(color[:, :, k],
                        extent=[xmin, xmax, ymin, ymax],
                        vmin=0, vmax=1, cmap=cmap, origin='lower')
        ax.set_aspect('auto')
    ax.set_title('true vector field')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

# In[]
    #Plot inferred vector field and color by discrete state
    ax = fig.add_subplot(gs[1, 1])
    X, Y, arrows = plotting.rot_vector_field(At[-1], trslds.R, xmin, xmax, ymin, ymax, delta, trslds.depth,
                                             trslds.leaf_paths,
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
    ax.set_title('inferred vector field')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

# In[]
    #Plot vector field of root node
    ax = fig.add_subplot(gs[0, 2])
    root_depth = 1
    root_K = 1
    _, root_leaf_path, _, _ = utils.create_balanced_binary_tree(root_K)
    X, Y, arrows = plotting.rot_vector_field(At[0], trslds.R, xmin, xmax, ymin, ymax, delta, root_depth,
                                             root_leaf_path, root_K, transform)
    norm = np.sqrt(arrows[:, :, 0] ** 2 + arrows[:, :, 1] ** 2)
    U = arrows[:, :, 0] / norm
    V = arrows[:, :, 1] / norm

    ax.streamplot(X, Y, U, V, color=np.log(norm), cmap='plasma_r')
    ax.set_title('vector field of root node')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')


# In[]
    #Plot vector field of second layer
    ax = fig.add_subplot(gs[1, 2])
    second_depth = 2
    second_K = 2
    _, second_leaf_path, _, _ = utils.create_balanced_binary_tree(second_K)
    X, Y, arrows = plotting.rot_vector_field(At[1], trslds.R, xmin, xmax, ymin, ymax, delta, second_depth,
                                             second_leaf_path, second_K, transform)
    norm = np.sqrt(arrows[:, :, 0] ** 2 + arrows[:, :, 1] ** 2)
    U = arrows[:, :, 0] / norm
    V = arrows[:, :, 1] / norm

    ax.streamplot(X, Y, U, V, color=np.log(norm), cmap='plasma_r')

    X, Y, color = plotting.rot_contour_plt(trslds.R, xmin, xmax, ymin, ymax, delta, second_depth, second_leaf_path,
                                           second_K, transform)

    for k in range(second_K):
        start = np.array([1., 1., 1., 0.])
        end = np.concatenate((colors_leaf[k], [0.5]))
        cmap = plotting.gradient_cmap([start, end])
        im1 = ax.imshow(color[:, :, k],
                        extent=[xmin, xmax, ymin, ymax],
                        vmin=0, vmax=1, cmap=cmap, origin='lower')
        ax.set_aspect('auto')
    ax.set_title('vector field of second layer')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    fig.tight_layout()
    fig.show()

