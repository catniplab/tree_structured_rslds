import copy
import numpy as np
import numpy.random as npr
import bayes_leaf as samp
from scipy import linalg
from tqdm import tqdm
import stuff


def gibbs_leaf_sampler(y, u, scale, dim, K, NoSamples, nu, nuy,\
                       Lambda_x, Lambda_y, My, Vy, mu_prior, \
                       Sigma_prior, Mx, Vx, Ux, A, B, C, D, R, r, x, z, path, Q,
                       S, possible_paths, leaf_path, leaf_nodes, prior_cov, bern, hack):
    """
    y= data. The variable is a 3D array where each slice corresponds to a different time series
    scale= User determined hyper parameter that scales down the covariance matrix (scales up your precision matrix) as you go deeper down the tree.
    dim= dimension of the latent space
    start= starting point for each of the time series
    K = number of leaf nodes
    NoSamples= Number of samples obtained from MCMC
    max_epochs, batch_size, lr = parameters for MSE initialization
    """
    dim_y = y[:, 0, 0].size  # Find dimension of observed data
    dim_input = u[:, 0, 0].size
    T = y[0, :, 0].size  # Length of time series
    num_trajec = y[0, 0, :].size  # Number of time series
    temper = 1  # parameter in sigmoid function used to make decision boundaries sharper
    depth = int(np.ceil(np.log2(K)) + 1)  # Find the maximum depth of the tree

    cutoff = int(0.01 * z.size)  # Cutoff point that lets the sampler know to rescue the branch
    resample = False  # Flag for knowing when to resample discrete latent states

    # Sample Polya-Gamma rvs
    w = np.zeros((depth, T + 1, num_trajec))
    for trajec in range(num_trajec):
        w[:, :, trajec] = samp.PG_tree_posterior(x[:, :, trajec], w[:, :, trajec], R, r, path[:, :, trajec], depth)

    if bern:
        # Sample PG for spike trains
        wy = np.zeros((dim_y, T, num_trajec))
        for trajec in range(num_trajec):
            wy[:, :, trajec] = samp.PG_spike_train(y[:, :, trajec], x[:, 1:, trajec],
                                                   C, D)

    # Create variables for Kalman Filter
    P = 10 * np.eye(dim)
    P = np.repeat(P[:, :, np.newaxis], T + 1, axis=2)
    alphas = np.zeros((dim, T))
    Lambdas = 10 * np.eye(dim)
    Lambdas = np.repeat(Lambdas[:, :, np.newaxis], T, axis=2)

    x_samples = []
    x_samples.append(copy.deepcopy(x))

    path_samples = []
    path_samples.append(copy.deepcopy(path))

    z_samples = []
    z_samples.append(copy.deepcopy(z))

    R_samples = []
    if depth != 1:
        R_samples.append(copy.deepcopy(R))

    r_samples = []
    if depth != 1:
        r_samples.append(copy.deepcopy(r))

    A_samples = []
    A_samples.append(copy.deepcopy(A))

    B_samples = []
    B_samples.append(copy.deepcopy(B))

    C_samples = []
    C_samples.append(copy.deepcopy(C))

    D_samples = []
    D_samples.append(copy.deepcopy(D))

    if bern:
        wy_samples = []
        wy_samples.append(copy.deepcopy(wy))
    else:
        S_samples = []
        S_samples.append(copy.deepcopy(S))

    Q_samples = []
    Q_samples.append(copy.deepcopy(Q))

    w_samples = []
    w_samples.append(copy.deepcopy(w))

    bad_indices = []

    def compute_leaf_dynamics():
        Aleaf = np.zeros((dim, dim, K))
        Bleaf = np.zeros((dim, dim_input, K))
        for (d, node, k) in leaf_nodes:
            Aleaf[:, :, k] = A[d][:, :, node]
            Bleaf[:, :, k] = B[d][:, :, node]

        return Aleaf, Bleaf

    """
    Start of Gibbs sampler
    """
    for m in tqdm(range(NoSamples)):

        if bern:
            "Obtain sample from emission conditional posterior."
            Emission_params = samp.emission_parameters_spike_train(copy.deepcopy(y), copy.deepcopy(x[:, 1:, :]),
                                                                   wy, np.matrix(My), np.matrix(Vy))
        else:
            "Obtain sample from emission conditional posterior."
            Emission_params, S = samp.emission_parameters(copy.deepcopy(y), copy.deepcopy(x[:, 1:, :]), nuy,
                                                          np.matrix(Lambda_y), np.matrix(My), np.matrix(Vy))
            S_samples.append(copy.deepcopy(S))

        C = Emission_params[:, :-1]
        D = Emission_params[:, -1]
        "Take RQ decomposition of C"
        upper, orthor = linalg.rq(C)
        upper = np.matrix(upper)
        orthor = np.matrix(orthor)

        "Constrain minor diagnoal of upper to be positive to avoid sign flipping"
        rotate = np.matrix(np.eye(dim))
        for j in range(dim):
            if np.sign(upper[dim_y - dim + j, j]) < 0:
                rotate[j, j] = -1

        upper = upper * rotate
        orthor = rotate * orthor

        "Rotate latent states and dynamics"
        x = stuff.rotate_latent(x, orthor)
        A, B = stuff.rotate_dynamics(A, B, orthor, depth)
        C = upper
        C_samples.append(copy.deepcopy(C))
        D_samples.append(copy.deepcopy(D))

        "Obtain sample from hyper planes conditional posterior"
        if depth != 1:
            R, r = stuff.sample_hyperplanes(x, w, path, depth, temper, mu_prior, Sigma_prior,
                                            possible_paths, num_trajec, R, r)



        "Obtain sample from LDS posterior of the leaf nodes"
        A, B, Q = stuff.sample_leaf_dynamics2(x, u, dim_input, z, A, B, Q, nu, Lambda_x, Mx, prior_cov, scale, leaf_nodes,
                                             num_trajec)
        # A, B, Q = stuff.sample_leaf_dynamics(x, u, dim_input, z, A, B, Q, nu, Lambda_x, Mx, Ux, scale, leaf_nodes,
        #                                     num_trajec)
        "Obtain sample from LDS posteriors of the interior nodes"
        A, B = stuff.sample_interior_dynamics2(A, B, prior_cov, scale, Mx, depth, dim_input)

        "Learn prior covaraince of dynamics"
        prior_cov = stuff.sample_prior_cov(A, B, Mx, scale, depth, prior_cov)
        # A, B = stuff.sample_interior_dynamics(A, B, scale, Mx, Vx, depth, dim_input)
        # A, B, Q = stuff.sample_leaf_dynamics(x, u, dim_input, z, A, B, Q, nu, Lambda_x, Mx, Ux, scale, leaf_nodes,
        #                                      num_trajec)
        Aleaf, Bleaf = compute_leaf_dynamics()

        "Obtain z_0:T from conditional posterior"
        if depth != 1:
            for trajec in range(num_trajec):
                tempz, temp_path = samp.discrete_latent(z[trajec, :], path[:, :, trajec], T + 1, leaf_path, K,
                                                        x[:, :, trajec], u[:, :, trajec], Aleaf, Bleaf, Q, R, r, depth)
                z[trajec, :] = tempz
                path[:, :, trajec] = temp_path

        "Obtain samples frpm pg conditional posterior"
        for trajec in range(num_trajec):
            w[:, :, trajec] = samp.PG_tree_posterior(x[:, :, trajec], w[:, :, trajec], R, r, path[:, :, trajec], depth)

        "Check to make sure all clusters are playing a role"
        if hack and depth != 1:
            for (d, node, k) in leaf_nodes:
                if np.sum(z == k) <= cutoff:
                    print('hi')
                    resample = True
                    # Find your parent
                    par_node = int(leaf_path[d - 1, k] - 1)

                    # Sample from hacky prior
                    temp_mu = 100000 * np.ones(dim + 1)
                    temp_mu[-1] = 0
                    temp_cov = 200000 * np.eye(dim + 1)
                    temp_cov[-1, -1] = 0.0001
                    Gamma = npr.multivariate_normal(temp_mu.flatten(), temp_cov)
                    R[-1][:, par_node] = Gamma[:-1]
                    r[-1][par_node] = Gamma[-1]

                    # Copy over your siblings dynamics
                    if k % 2 == 1:
                        A[d][:, :, node] = A[d][:, :, k - 1]
                        B[d][:, :, node] = B[d][:, :, k - 1]
                    else:
                        A[d][:, :, node] = A[d][:, :, k + 1]
                        B[d][:, :, node] = B[d][:, :, k + 1]

            if resample:
                resample = False
                bad_indices.append(m)
                Aleaf, Bleaf = compute_leaf_dynamics()
                # Resample discrete latent states
                if depth != 1:
                    for trajec in range(num_trajec):
                        tempz, temp_path = samp.discrete_latent(z[trajec, :], path[:, :, trajec], T + 1, leaf_path, K,
                                                                x[:, :, trajec], u[:, :, trajec], Aleaf, Bleaf, Q, R, r,
                                                                depth)
                        z[trajec, :] = tempz
                        path[:, :, trajec] = temp_path

                # Resample PG rvs
                for trajec in range(num_trajec):
                    w[:, :, trajec] = samp.PG_tree_posterior(x[:, :, trajec], w[:, :, trajec], R, r, path[:, :, trajec],
                                                             depth)

                "Obtain sample from LDS posteriors of the interior nodes"
                A, B = stuff.sample_interior_dynamics(A, B, scale, Mx, Vx, depth, dim_input)
                "Obtain sample from LDS posterior of the leaf nodes"
                A, B, Q = stuff.sample_leaf_dynamics(x, u, dim_input, z, A, B, Q, nu, Lambda_x, Mx, Ux, scale,
                                                     leaf_nodes, num_trajec)
                Aleaf, Bleaf = compute_leaf_dynamics()

                # Resample hyperplanes
                if depth != 1:
                    R, r = stuff.sample_hyperplanes(x, w, path, depth, temper, mu_prior, Sigma_prior,
                                                    possible_paths, num_trajec, R, r)

        if depth != 1:
            R_samples.append(copy.deepcopy(R))
            r_samples.append(copy.deepcopy(r))

        A_samples.append(copy.deepcopy(A))
        B_samples.append(copy.deepcopy(B))
        Q_samples.append(copy.deepcopy(Q))

        "Obtain x_0:T from conditional posterior"
        for trajec in range(num_trajec):
            if bern:
                "Obtain samples from pg spike train conditional posterior"
                wy[:, :, trajec] = samp.PG_spike_train(y[:, :, trajec], x[:, 1:, trajec], C, D)
                "Sample from x_0:T from conditional posterior"
                x[:, :, trajec] = samp.PG_KF_spike(dim, x[:, :, trajec], u[:, :, trajec], P, Aleaf, Bleaf, Q, C, D, T,
                                                   y[:, :, trajec], path[:, :, trajec],
                                                   z[trajec, :], w[:, :, trajec], wy[:, :, trajec], alphas, Lambdas, R,
                                                   r, depth)

            else:
                x[:, :, trajec] = samp.PG_KF(dim, x[:, :, trajec], u[:, :, trajec], P, Aleaf, Bleaf, Q, C, D, S, T,
                                             y[:, :, trajec], path[:, :, trajec], z[trajec, :], w[:, :, trajec],
                                             alphas, Lambdas, R, r, depth)

        x_samples.append(copy.deepcopy(x))
        path_samples.append(copy.deepcopy(path))
        w_samples.append(copy.deepcopy(w))
        z_samples.append(copy.deepcopy(z))

        if bern:
            wy_samples.append(copy.deepcopy(wy))

    if bern:
        estimates = {'A': A_samples, 'B': B_samples, 'Q': Q_samples, 'R': R_samples, 'r': r_samples, 'x': x_samples,
                     'C': C_samples, 'D': D_samples, 'wy': wy_samples, 'w': w_samples, 'path': path_samples,
                     'z': z_samples, 'sigma': prior_cov,
                     'leaf_path': leaf_path, 'leaf_nodes': leaf_nodes, 'idx': bad_indices}
    else:
        estimates = {'A': A_samples, 'B': B_samples, 'Q': Q_samples, 'R': R_samples, 'r': r_samples, 'x': x_samples,
                     'C': C_samples, 'D': D_samples, 'S': S_samples, 'w': w_samples, 'path': path_samples,
                     'z': z_samples, 'sigma': prior_cov,
                     'leaf_path': leaf_path, 'leaf_nodes': leaf_nodes, 'idx': bad_indices}
    return estimates