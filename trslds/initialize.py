import numpy as np
from scipy import linalg
from trslds import fit_greedy_mse as fit
from sklearn.decomposition import PCA
from trslds import utils
from numpy import newaxis as na
import numpy.random as npr

def initialize(Y, D_in, K, max_epochs, batch_size, lr, starting_pts=None):
    D_out = Y[0][:, 0].size  # Find dimension of observed data

    if starting_pts is None:
        starting_pts = 10*np.random.normal(size=(D_in, len(Y)))
    # Create balanced binary tree with K leaves
    depth, leaf_path, possible_paths, leaf_nodes = utils.create_balanced_binary_tree(K)

    "Initialization of emission parameters and continuous latent states"
    # Perform probabilistic PCA to get an estimate of the continuous latent states and the emission parameters
    tempy = np.hstack(Y).T
    model = PCA(n_components=D_in, whiten=False)
    tempx = model.fit_transform(tempy).T
    C = model.components_.T
    D = model.mean_[:, None]

    "Perform rq decomposition of C to remove rotation of states"
    upper, orthor = linalg.rq(C)
    rotate = np.eye(D_in)

    "Prevent sign flipping"
    for j in range(D_in):
        if np.sign(upper[D_out - D_in + j, j]) < 0:
            rotate[j, j] = -1

    upper = upper @ rotate
    orthor = rotate @ orthor

    "Rotate estimated latent states"
    tempx = orthor @ tempx

    C = upper  # initialize the emission matrix C
    C = np.hstack((C, D))  # Affine term is appended to last column of emission parameter

    # Format X correctly
    start = 0
    X = []
    for idx in range(len(Y)):
        fin = start + Y[idx][0, :].size
        X.append(tempx[:, start:fin])
        start = fin

    "Initialization of dynamic parameters, hyper-planes, discrete latent states"
    # To initialize this complex model, we first fit a similar version where the goal is to minimize MSE
    print("Initialization")
    LDS_init, nu_init = fit.initialize_dynamics(X, depth, max_epochs, batch_size, lr)
    print("End of Initialization")

    "Append starting points to time series"
    for idx in range(len(Y)):
        X[idx] = np.hstack((starting_pts[:, idx][:, na], X[idx])) + 0*npr.multivariate_normal(np.zeros(D_in), 0.1*np.eye(D_in),
         size = X[idx][0, :].size + 1).T

    "Initialize the dynamics of the tree"
    # Dynamic Parameters
    A = [None] * depth
    # Hyper planes
    R = [None] * (depth - 1)

    "Initializing dynamic parameters and hyper planes using values obtained from MSE initialization"
    for d in range(depth):
        """
        Allocate temporary memory for storing parameters
        """
        A_t = np.zeros((D_in, D_in + 1, 2 ** int(d)))
        R_t = np.zeros((D_in + 1, 2 ** int(d)))

        for node in range(2 ** int(d)):
            """
            Initalize with values obtained from LS version
            """
            if np.isnan(possible_paths[d, node]) != True:
                A_t[:, :-1, node] = LDS_init[d][:, :-1, node] + np.eye(D_in)
                A_t[:, -1, node] = LDS_init[d][:, -1, node]

                if d != 0:
                    A_t[:, :-1, node] += A[d - 1][:, :-1, int(np.floor(node / 2))] - np.eye(D_in)
                    A_t[:, -1, node] += A[d - 1][:, -1, int(np.floor(node / 2))]
            else:
                A_t[:, :, node] = np.nan * np.ones((D_in, D_in + 1))

            if d != depth - 1:
                R_t[:, node] = nu_init[d][:, node]

                if np.isnan(possible_paths[d + 1, 2 * node + 1]):
                    R_t[:, node] = np.nan

        A[d] = A_t

        if d != depth - 1:
            R[d] = R_t

    "Initalizing paths taken"
    Z, Path = fit.initialize_discrete(X, R, depth, K, leaf_path)

    return A, C, R, X, Z, Path, possible_paths, leaf_path, leaf_nodes
