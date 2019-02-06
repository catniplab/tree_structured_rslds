#Will be used for small stuff that can be parallelized
import numpy as np
import numpy.random as npr
import numba
from numba import jit
import bayes_leaf as samp
import copy
from joblib import Parallel, delayed

"Rotate x_0:T"
def rotate_latent(x, orthor):
    num_trajec = x[0,0,:].size
    for trajec in range(num_trajec): #parallelize rotations
        temp = np.matrix(x[:, :, trajec])
        x[:, :, trajec] = orthor * temp

    return x

"Rotate dynamics"
def rotate_dynamics(A, B, orthor, depth):

    for d in range(depth):
        for node in range(2 ** d):
            A[d][:, :, node] = orthor * np.matrix(A[d][:, :, node]) * orthor.T
            B[d][:, :, node] = orthor * np.matrix(B[d][:, :, node])
    return A, B


"Prime for the sampling of hyper-planes"
#@jit(nopython=True, parallel=True)
def sample_hyperplanes(x, w, path, depth, temper, mu_prior, Sigma_prior, possible_paths, num_trajec, R, r):
    dim = x[:,0,0].size
    for d in range(depth - 1):
        x_temp = x
        for node in range(2**d):
            effective_x = []
            effective_w = []
            effective_z = []
            min_size = 0
            # Check to see if current node is a leaf node or not
            if np.isnan(possible_paths[d + 1, 2 * node + 1]) == False:
                for trajec in range(num_trajec):
                    indices = (path[d, :, trajec] == (node + 1))
                    if x_temp[:, indices, trajec].size != 0:
                        effective_x.append(copy.deepcopy(x_temp[:, indices, trajec]))
                        effective_w.append(copy.deepcopy(w[d, indices, trajec]))
                        effective_z.append(copy.deepcopy(path[d + 1, indices, trajec]))
                        min_size += x_temp[0, indices, trajec].size

                draw_prior = min_size == 0  # Boolean variable that will indicate if we should draw from the prior or not
                temp_R, temp_r = samp.hyper_planes(copy.deepcopy(effective_w), copy.deepcopy(effective_x),
                                                   copy.deepcopy(effective_z), mu_prior, Sigma_prior,
                                                   temper, draw_prior, dim)
                R[d][:, node] = copy.deepcopy(temper * temp_R)
                r[d][node] = copy.deepcopy(temper * temp_r)
    return R,r

"Sample a specific hyper-plane"
def specific_hyperplane(x, w, path, level, node, temper, mu_prior, Sigma_prior, possible_paths, num_trajec):
    effective_x = []
    effective_w = []
    effective_z = []
    dim = x[:, 0, 0].size

    for trajec in range(num_trajec):
        indices = (path[level, :, trajec] == (node + 1))
        if x[:, indices, trajec].size != 0:
            effective_x.append(copy.deepcopy(x[:, indices, trajec]))
            effective_w.append(copy.deepcopy(w[level, indices, trajec]))
            effective_z.append(copy.deepcopy(path[level + 1, indices, trajec]))


    temp_R, temp_r = samp.hyper_planes(copy.deepcopy(effective_w), copy.deepcopy(effective_x),
                                       copy.deepcopy(effective_z), mu_prior, Sigma_prior,
                                       temper, False, dim)
    return temp_R, temp_r


"Sample prior covariances of dynamics"
def sample_prior_cov(A, B, Mx, scale, depth, priorcovs):
    for d in range(depth):
        for node in range(int(2**d)):
            # Check to see if current node is a leaf node
            if not np.isnan(A[d][0, 0, node]):
                if d == 0:
                    parent = Mx
                else:
                    parent = np.hstack(
                        (A[d - 1][:, :, int(np.floor(node / 2))], np.matrix(B[d - 1][:, :, int(np.floor(node / 2))])))/np.sqrt(scale**d)
                child = np.hstack((A[d][:, :, node],
                                   np.matrix(B[d][:, :, node])))/np.sqrt(scale**d)
                priorcovs[d][node] = samp.compute_prior_covariance(child, parent)
    return priorcovs

"sample interior dynamics"
def sample_interior_dynamics2(A, B, priorcovs, scale, Mx, depth, dim_input):
    # for d in range(depth - 1):
    for d in range(depth - 2, -1, -1):
        for node in range(int(2 ** d)):
            # Check to see if current node is a leaf node
            if not np.isnan(A[d + 1][0, 0, 2 * node]):
                if d == 0:
                    Mprior = Mx
                else:
                    Mprior = np.hstack(
                        (A[d - 1][:, :, int(np.floor(node / 2))], np.matrix(B[d - 1][:, :, int(np.floor(node / 2))])))
                tau_prior = priorcovs[d][node] * scale**d
                
                # if current node isn't a leaf node
                child1 = np.hstack((A[d + 1][:, :, 2 * node], 
                                    B[d + 1][:, :, 2 * node]))
                tau1 = priorcovs[d + 1][2 * node] * scale**(d + 1)
                
                child2 = np.hstack((A[d + 1][:, :, 2 * node + 1], 
                                    B[d + 1][:, :, 2 * node + 1]))
                tau2 = priorcovs[d + 1][2 * node + 1]* scale**(d + 1)
                
                sampleA, sampleB = samp.interior_dynamics_pt2(child1, tau1, child2, tau2, Mprior, tau_prior, dim_input)
                A[d][:, :, node] = sampleA
                B[d][:, :, node] = sampleB
    return A, B

def sample_leaf_dynamics2(x, u, dim_input, z, A, B, Q, nu, Lambda_x, Mx, priorcovs, scale, leaf_nodes, num_trajec ):
    for (d, node, k) in leaf_nodes:
        eff_x = []
        eff_y = []
        eff_u = []
        psuedo_x = x[:, :-1, :]
        psuedo_u = u[:, :-1, :]
        psuedo_y = x[:, 1:, :]
        psuedo_z = z[:, :-1]
        min_size = 0
        for trajec in range(num_trajec):
            idx = psuedo_z[trajec, :] == k
            if psuedo_x[:, idx, trajec].size != 0:
                eff_x.append(copy.deepcopy(psuedo_x[:, idx, trajec]))
                eff_u.append(copy.deepcopy(psuedo_u[:, idx, trajec]))
                eff_y.append(copy.deepcopy(psuedo_y[:, idx, trajec]))
                min_size += psuedo_x[0, idx, trajec].size

        if d == 0:
            Mprior = Mx
        else:
            Mprior = np.hstack(
                (A[d - 1][:, :, int(np.floor(node / 2))], np.matrix(B[d - 1][:, :, int(np.floor(node / 2))])))

        draw_prior = min_size == 0
        Ux = priorcovs[d][node] * np.eye(dim_input + x[:, 0, 0].size)
        sampleA, sampleB, sampleQ = samp.leaf_dynamics(eff_x, eff_u, dim_input, eff_y,  Mprior, (scale ** d) * Ux, nu, Lambda_x,
                                                       draw_prior)
        A[d][:, :, int(node)] = sampleA
        B[d][:, :, int(node)] = sampleB
        Q[:, :, int(k)] = sampleQ
    return A, B, Q

"Prime for sampling of interior dynamics"
def sample_interior_dynamics(A, B, scale, Mx, Vx, depth, dim_input):
    for d in range(depth - 1):
        for node in range(int(2 ** d)):
            A_child = 0
            B_child = 0

            # Check to see if current node is a leaf node
            if not np.isnan(A[d + 1][0, 0, 2 * node]):
                if d == 0:
                    Mprior = Mx
                else:
                    Mprior = np.hstack(
                        (A[d - 1][:, :, int(np.floor(node / 2))], np.matrix(B[d - 1][:, :, int(np.floor(node / 2))])))
                # if current node isn't a leaf node
                for g in range(2):
                    A_child += A[d + 1][:, :, 2 * node + g]
                    B_child += B[d + 1][:, :, 2 * node + g]

                # obtain sample from posterior
                sampleA, sampleB = samp.interior_dynamics(A_child, B_child, (scale ** (d + 1)) * Vx, Mprior,
                                                          (scale ** d) * Vx, dim_input)
                A[d][:, :, node] = sampleA
                B[d][:, :, node] = sampleB
    return A, B

"Prime sampling from the LDS posterior of leaf nodes"
def sample_leaf_dynamics(x, u, dim_input, z, A, B, Q, nu, Lambda_x, Mx, Ux, scale, leaf_nodes, num_trajec ):
    for (d, node, k) in leaf_nodes:
        eff_x = []
        eff_y = []
        eff_u = []
        psuedo_x = x[:, :-1, :]
        psuedo_u = u[:, :-1, :]
        psuedo_y = x[:, 1:, :]
        psuedo_z = z[:, :-1]
        min_size = 0
        for trajec in range(num_trajec):
            idx = psuedo_z[trajec, :] == k
            if psuedo_x[:, idx, trajec].size != 0:
                eff_x.append(copy.deepcopy(psuedo_x[:, idx, trajec]))
                eff_u.append(copy.deepcopy(psuedo_u[:, idx, trajec]))
                eff_y.append(copy.deepcopy(psuedo_y[:, idx, trajec]))
                min_size += psuedo_x[0, idx, trajec].size

        if d == 0:
            Mprior = Mx
        else:
            Mprior = np.hstack(
                (A[d - 1][:, :, int(np.floor(node / 2))], np.matrix(B[d - 1][:, :, int(np.floor(node / 2))])))

        draw_prior = min_size == 0
        sampleA, sampleB, sampleQ = samp.leaf_dynamics(eff_x, eff_u, dim_input, eff_y,  Mprior, (scale ** d) * Ux, nu, Lambda_x,
                                                       draw_prior)
        A[d][:, :, int(node)] = sampleA
        B[d][:, :, int(node)] = sampleB
        Q[:, :, int(k)] = sampleQ
    return A, B, Q

"Prime sampling from the LDS posterior of leaf nodes"
def sample_leaf_dynamics_annealed(x, u, dim_input, z, A, B, Q, nu, Lambda_x, Mx, Ux, scale, leaf_nodes, num_trajec, beta):
    for (d, node, k) in leaf_nodes:
        eff_x = []
        eff_y = []
        eff_u = []
        psuedo_x = x[:, :-1, :]
        psuedo_u = u[:, :-1, :]
        psuedo_y = x[:, 1:, :]
        psuedo_z = z[:, :-1]
        min_size = 0
        for trajec in range(num_trajec):
            idx = psuedo_z[trajec, :] == k
            if psuedo_x[:, idx, trajec].size != 0:
                eff_x.append(copy.deepcopy(psuedo_x[:, idx, trajec]))
                eff_u.append(copy.deepcopy(psuedo_u[:, idx, trajec]))
                eff_y.append(copy.deepcopy(psuedo_y[:, idx, trajec]))
                min_size += psuedo_x[0, idx, trajec].size

        if d == 0:
            Mprior = Mx
        else:
            Mprior = np.hstack(
                (A[d - 1][:, :, int(np.floor(node / 2))], np.matrix(B[d - 1][:, :, int(np.floor(node / 2))])))

        draw_prior = min_size == 0
        sampleA, sampleB, sampleQ = samp.leaf_dynamics_annealed(eff_x, eff_u, dim_input, eff_y,  Mprior, (scale ** d) * Ux, nu, Lambda_x,
                                                       draw_prior, beta)
        A[d][:, :, int(node)] = sampleA
        B[d][:, :, int(node)] = sampleB
        Q[:, :, int(k)] = sampleQ
    return A, B, Q

########################################################################################################################
"Parallel sampling for discrete states"
def sample_discrete_states(z, path, x, u, y, w, leaf_path, K, Aleaf, Bleaf, Q, R, r, depth):
    z, path = samp.discrete_latent(z, path, x[0,:].size, leaf_path, K, x, u, Aleaf, Bleaf, Q, R, r, depth)
    return x, u, z, path, y, w


def parallel_discrete_sampling(z, path, x, u, y, w, num_trajec,leaf_path, K, Aleaf, Bleaf, Q, R, r, depth):

    returns = Parallel(n_jobs = -1)( delayed(sample_discrete_states)(z[trajec,:], path[:,:,trajec], x[:,:,trajec],
                                                                    u[:,:,trajec], y[:,:,trajec], w[:,:,trajec], leaf_path, K, Aleaf, Bleaf, Q, R,
                                                                     r, depth) for trajec in range(num_trajec))

    for trajec in range(num_trajec):
        x[:, :, trajec] = returns[trajec][0]
        u[:, :, trajec] = returns[trajec][1]
        z[trajec, :] = returns[trajec][2]
        path[:, :, trajec] = returns[trajec][3]
        y[:, :, trajec] = returns[trajec][4]
        w[:, :, trajec] = returns[trajec][5]
    return x, u, z, path, y, w
########################################################################################################################




########################################################################################################################
"Parallel sampling for continuous states with gaussian observations"
def sample_latent_states_gauss(x, u, P, dim, Aleaf, Bleaf, Q, C, D, S, T, y, path, z, w,
                         alphas, Lambdas, R, r, depth):
    x = samp.PG_KF(dim, x, u, P, Aleaf, Bleaf, Q, C, D, S, T,
                   y, path[:, :-1], z, w[:, :-1], alphas, Lambdas, R, r, depth)

    return x, u, z, path, y, w


def parallel_continuous_sampling_gauss(x, u, P, dim, Aleaf, Bleaf, Q, C, D, S, T, y, path, z, w,
                         alphas, Lambdas, R, r, depth, num_trajec):

    returns = Parallel(n_jobs=-1)(
        delayed(sample_latent_states_gauss)(x[:,:,trajec], u[:,:,trajec], P, dim, Aleaf, Bleaf, Q, C, D, S, T, y[:,:,trajec],
                                      path[:,:,trajec], z[trajec, :], w[:,:,trajec], alphas, Lambdas, R, r, depth)
        for trajec in range(num_trajec))

    for trajec in range(num_trajec):
        x[:, :, trajec] = returns[trajec][0]
        u[:, :, trajec] = returns[trajec][1]
        z[trajec, :] = returns[trajec][2]
        path[:, :, trajec] = returns[trajec][3]
        y[:, :, trajec] = returns[trajec][4]
        w[:, :, trajec] = returns[trajec][5]

    return x, u, z, path, y, w
########################################################################################################################


########################################################################################################################
"Parallel sampling for continuous states with gaussian observations"
def sample_latent_states_bern(x, u, P, dim, Aleaf, Bleaf, Q, C, D, T, y, path, z, w,
                         alphas, Lambdas, R, r, depth):
    #Sample PG rvs for spike train
    wy = samp.PG_spike_train(y, x[:,1:], C, D)
    #Sample continuous latent states
    x = samp.PG_KF_spike(dim, x, u[:,:-1], P, Aleaf, Bleaf, Q, C, D, T,
                   y, path[:, :-1], z, w[:, :-1], wy, alphas, Lambdas, R, r, depth)

    return x, u, z, path, y, w, wy


def parallel_continuous_sampling_bern(x, u, P, dim, Aleaf, Bleaf, Q, C, D, T, y, path, z, w, wy,
                         alphas, Lambdas, R, r, depth, num_trajec):

    returns = Parallel(n_jobs=-1)(
        delayed(sample_latent_states_bern)(x[:,:,trajec], u[:,:,trajec], P, dim, Aleaf, Bleaf, Q, C, D, T, y[:,:,trajec],
                                      path[:, :, trajec], z[trajec,:], w[:,:,trajec], alphas, Lambdas,
                                           R, r, depth) for trajec in range(num_trajec))
    for trajec in range(num_trajec):
        x[:, :, trajec] = returns[trajec][0]
        u[:, :, trajec] = returns[trajec][1]
        z[trajec, :] = returns[trajec][2]
        path[:, :, trajec] = returns[trajec][3]
        y[:, :, trajec] = returns[trajec][4]
        w[:, :, trajec] = returns[trajec][5]
        wy[:, :, trajec] = returns[trajec][6]
    return x, u, z, path, y, w, wy
########################################################################################################################
    

########################################################################################################################
def sample_pg(x, u, y, w, R, r, z, path, depth):
    w = samp.PG_tree_posterior(x, w, R, r, path, depth)
    return x, u, z, path, y, w

def parallel_pg(x, u, y, w, R, r, z, path, depth, num_trajec):
    returns = Parallel(n_jobs=-1)( delayed(sample_pg)(x[:, :, trajec], u[:, :, trajec], y[:,:,trajec],
                       w[:,:,trajec], R, r, z[trajec,:], path[:, :, trajec], depth) 
                for trajec in range(num_trajec))
    for trajec in range(num_trajec):
        x[:, :, trajec] = returns[trajec][0]
        u[:, :, trajec] = returns[trajec][1]
        z[trajec, :] = returns[trajec][2]
        path[:, :, trajec] = returns[trajec][3]
        y[:, :, trajec] = returns[trajec][4]
        w[:, :, trajec] = returns[trajec][5]

    return x, u, z, path, y, w
########################################################################################################################
