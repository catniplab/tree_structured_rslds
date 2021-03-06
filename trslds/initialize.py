import numpy as np
from trslds import fit_greedy_mse as fit
from sklearn.decomposition import PCA
from . import utils
from numpy import newaxis as na
import numpy.random as npr
import matplotlib.pyplot as plt


def initialize(Y, dx, K, max_epochs, batch_size, lr, X=None, u=None, random=False, plot=False):
    dy = Y[0][:, 0].size  # Find dimension of observed data

    if u is None:
        du = 1
        u = [np.ones((1, Y[idx][0, :].size - 1)) for idx in range(len(Y))]
    else:
        du = u[0][:, 0].size
        u = [u[idx][:, 1:] for idx in range(len(u))]

    # Create balanced binary tree with K leaves
    depth, leaf_path, possible_paths, leaf_nodes = utils.create_balanced_binary_tree(K)
    tempy = np.hstack(Y).T
    if X is None:
        "Initialization of emission parameters and continuous latent states"
        # Perform probabilistic PCA to get an estimate of the continuous latent states and the emission parameters
        model = PCA(n_components=dx, whiten=True)
        tempx = model.fit_transform(tempy).T
        C = model.components_.T
        D = model.mean_[:, None]
    else:
        tempx = np.hstack(X).T
        # z-score latents
        tempx -= np.mean(tempx, 1)[:, na]
        tempx /= np.std(tempx, 1)[:, na]
        x_ols = np.hstack((tempx.T, (np.ones((tempx[0, :].size, 1)))), 1)
        beta = np.linalg.solve((x_ols.T @ x_ols), x_ols.T @ tempy.T).T
        C = beta[:, :-1]
        D = beta[:, -1]

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
    LDS_init, nu_init, losses = fit.top_to_bottom(X, depth, max_epochs, batch_size, lr, u=u)
    print("End of Initialization")

    if plot:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(losses)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        fig.show()

    "Append starting points to time series"
    for idx in range(len(Y)):
        start_pt = X[idx][:, 0] + 0.1 * npr.randn(dx)
        X[idx] = np.hstack((start_pt[:, na], X[idx]))

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
        A_t = np.zeros((dx, dx + du, 2 ** int(d)))
        R_t = np.zeros((dx + 1, 2 ** int(d)))

        for node in range(2 ** int(d)):
            """
            Initialize with values obtained from LS version
            """
            if ~np.isnan(possible_paths[d, node]):
                A_t[:, :-du, node] = LDS_init[d][:, :-du, node]
                A_t[:, -du:, node] = LDS_init[d][:, -du:, node]

                if d != 0:
                    A_t[:, :-du, node] += A[d - 1][:, :-du, int(np.floor(node / 2))]
                    A_t[:, -du:, node] += A[d - 1][:, -du:, int(np.floor(node / 2))]
            else:
                A_t[:, :, node] = np.nan * np.ones((dx, dx + du))

            if d != depth - 1:
                R_t[:, node] = nu_init[d][:, node]

                if np.isnan(possible_paths[d + 1, 2 * node + 1]):
                    R_t[:, node] = np.nan

        A[d] = A_t

        if d != depth - 1:
            R[d] = R_t

    "Initializing paths taken"
    Z, Path = fit.initialize_discrete(X, R, depth, K, leaf_path, random=random)

    return A, C, R, X, Z, Path, possible_paths, leaf_path, leaf_nodes
