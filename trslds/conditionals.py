import numpy as np
from . import utils
from numpy import newaxis as na
import numpy.random as npr
import scipy
from numpy.linalg import LinAlgError
from scipy.stats import invwishart
from pypolyagamma import PyPolyaGamma
import pypolyagamma
from numba import jit
import pdb
from os import cpu_count

# In[1]:
def pg_tree_posterior(states, omega, R, path, depth, nthreads=None):
    '''
    Sample Polya-Gamma w_n,t|x_t,z_{t+1} where the subscript n denotes the hyperplane
    for which we are augmenting with the Polya-Gamma. Thus will augment all the logistic regressions
    that was taken while traversing down the tree
    :param states: This variable contains the continuous latent states. It is a list of numpy arrays
    :param omega: list for storing polya-gamma variables
    :param R: normal vectors of hyper-plane where the bias term is the last element in that array. The format is a list of arrays.
    :param path: path taken through the tree at time t. a list of numpy arrays
    :param depth: maximum depth of the tree
    :return: a list of pg rvs for each time series
    '''
    for idx in range(len(states)):
        T = states[idx][0, :].size
        b = np.ones(T * (depth - 1))
        if nthreads is None:
            nthreads = cpu_count()
        v = np.ones((depth - 1, T))
        out = np.empty(T * (depth - 1))
        #Compute parameters for conditional
        for d in range(depth - 1):
            for t in range(T):
                index = int(path[idx][d, t] - 1)  # Find which node you went through
                v[d, t] = np.matmul(R[d][:-1, index], np.array(states[idx][:, t])) + R[d][-1, index]
        seeds = np.random.randint(2 ** 16, size=nthreads)
        ppgs = [PyPolyaGamma(seed) for seed in seeds]
        #Sample in parallel
        pypolyagamma.pgdrawvpar(ppgs, b, v.flatten(order='F'), out)
        omega[idx] = out.reshape((depth - 1, T), order='F')

    return omega


# In[2]:
def pg_spike_train(X, C, Omega, D_out, nthreads=None):
    '''
    Sample Polya-Gamma wy|Y,C,D,X where Y are spike trains and X are the continuous latent states
    :param X: continuous latent states
    :param C: emission parameters. bias parameter is appended to last column.
    :param Omega: list used for storing polya-gamma variables
    :param D_out: Dimension of output i..e number of neurons
    :return: polya gamma samples from conditional posterior in a list of numpy arrays
    '''
    for idx in range(len(X)):
        T = X[idx][0, 1:].size
        b = np.ones(T * D_out)
        if nthreads is None:
            nthreads = cpu_count()
        out = np.empty(T * D_out)
        V = C[:, :-1] @ X[idx][:, 1:] + C[:, -1][:, na]  # Ignore the initial point of the time series

        seeds = np.random.randint(2 ** 16, size=nthreads)
        ppgs = [PyPolyaGamma(seed) for seed in seeds]

        pypolyagamma.pgdrawvpar(ppgs, b, V.flatten(order='F'), out)
        Omega[idx] = out.reshape((D_out, T), order='F')

    return Omega


# In[3]:
def emission_parameters(obsv, states, mask, nu, Lambda, M, V, normalize=True):
    '''
    Sampling from MNIW conditional distribution for emission parameters
    :param obsv: list of observations
    :param states: list of continuous latentstates
    :param mask: boolean mask used to remove missing data. A list of mask for each time series
    :param nu: prior degree of freedoms
    :param Lambda: prior on noise covariance
    :param M: prior mean of emission
    :param V: prior row covariance
    :param normalize: boolean variable that dictates whether to normalize the columns of C
    :return:  Sample from MNIW posterior, (C, S), where C is normalized to unit column norms for psuedo manifold sampling
    '''

    Y = np.hstack(obsv).T  # stack observations
    X = np.hstack([states[idx][:, 1:] for idx in range(len(states))])  # Stack observations disregarding the first point
    X = np.vstack((X, np.ones((1, X[0, :].size)))).T  # Stack vector of ones to learn affine term

    boolean_mask = np.hstack(mask).T
    # Get rid of missing observations
    Y = Y[boolean_mask, :]
    X = X[boolean_mask, :]

    M_posterior, V_posterior, IW_matrix, df_posterior = utils.compute_ss_mniw(X, Y, nu, Lambda, M, V)

    C, S = utils.sample_mniw(df_posterior, IW_matrix, M_posterior, V_posterior)

    if normalize: #If true then normalize columns of C (except the last column which is the affine term)
        C_temp = C[:, :-1]
        #Normalize columns
        L = np.diag(C_temp.T @ C_temp)
        L = np.diag(np.power(L, -0.5))
        C_temp = C_temp @ L
        C[:, :-1] = C_temp

    return C, S


# In[4]:
def emission_parameters_spike_train(spikes, states, Omega, mask, mu, Sigma, normalize=True):
    '''
    Sample from conditional posterior of emission parameters where emission model is a bernoulli glm.
    :param spikes: list of spike trains
    :param states: list of continuous latent states
    :param Omega: list of polya gamma rvs
    :param mask: boolean mask denoting missing data
    :param mu: prior mean
    :param Sigma: prior covariance
    :param normalize: boolean variable that dictates whether to normalize the columns of C
    :return: sample from conditional posterior, C where columns are normalized.
    '''
    X = np.hstack([states[idx][:, 1:] for idx in range(len(states))]) #stack all continuous latent states (ignore initial starting point)
    X = np.vstack((X, np.ones(X[0, :].size)))  #append a vecor of ones for affine term
    Y = np.hstack(spikes) #stack all spike trains
    W = np.hstack(Omega) #stack all spike polya-gamma rvs
    boolean_mask = np.hstack(mask) #stack boolena mask

    #Mask missing spike trains
    X = X[:, boolean_mask]
    Y = Y[:, boolean_mask]
    W = W[:, boolean_mask]
    
    dim_y = Y[:, 0].size
    dim = X[:-1, 0].size
    C = np.zeros((dim_y, dim + 1))

    for neuron in range(dim_y):
        Lambda_post = np.linalg.inv(Sigma)
        temp_mu = np.matmul(mu, Lambda_post)

        xw_tilde = np.multiply(X, np.sqrt(W[neuron, :]))  # pre multiply by sqrt(w_n,t )
        Lambda_post += np.einsum('ij,ik->jk', xw_tilde.T, xw_tilde.T)  # Use einstein summation to compute sum of outer products

        temp_mu += np.sum((X * (Y[neuron, :][na, :] - 0.5)).T, axis=0)
        Sigma_post = np.linalg.inv(Lambda_post)
        mu_post = np.matmul(temp_mu, Sigma_post)
        # Sample from mvn posterior
        C[neuron, :] = npr.multivariate_normal(np.array(mu_post).ravel(), Sigma_post)

    if normalize: #If true then normalize columns of C (except the last column which is the affine term)
        #Normalize columns of C
        C_temp = C[:, :-1]
        L = np.diag(np.matmul(C_temp.T, C_temp))
        L = np.diag(np.power(L, -0.5))
        C_temp = np.matmul(C_temp, L)
        C[:, :-1] = C_temp

    return C


# In[5]:
#@jit(nopython=True)
def hyper_planes(w, x, z, prior_mu, prior_precision, draw_prior):
    '''
    Sample from the conditional posterior of the hyperplane. Due to the polya-gamma augmentation, the model is normal.
    :param w: list of polya gamma rvs
    :param x: list of continuous latent states
    :param z: list of tree indices
    :param prior_mu: prior mean
    :param prior_sigma: prior covariance
    :param draw_prior: boolean variable indicating whether to sample from the prior or not
    :return: sample from conditional posterior
    '''
    if draw_prior == False:
        J = prior_precision @ prior_mu[:, na] # J = Sigma^{-1}*mu
        xw_tilde = np.multiply(x, np.sqrt(w[na, :]))  # pre multiply by sqrt(w_n,t )
        precision = np.einsum('ij,ik->jk', xw_tilde.T,
                                 xw_tilde.T)  # Use einstein summation to compute sum of outer products

        k = z % 2 - 0.5 #Check to see if you went left or right from current node
        J += np.sum(x * k[na, :], axis=1)[:, na]
        
        posterior_cov = np.linalg.inv(precision + prior_precision) #Obtain posterior covariance
        posterior_mu = posterior_cov @ J

        return npr.multivariate_normal(posterior_mu.flatten(), posterior_cov)  # Return sample from posterior.
    else:
        return npr.multivariate_normal(prior_mu, prior_precision)  # If no data points then draw from prior.

# In[6]:
def _internal_dynamics(Mprior, Vparent, Achild, Vchild, N=2):
    '''
    Sample from dynamics conditional of an internal node in tree
    :param Mprior: prior of dynamics
    :param Vparent: prior column covariance
    :param Achild: sum of realization of dynamics of children
    :param Vchild: children column covariance
    :return:
    '''
    assert Mprior.shape == Achild.shape
    precision_parent = np.linalg.inv(np.kron(Vparent, np.eye(Achild[:, 0].size)))
    precision_child = np.linalg.inv(np.kron(Vchild, np.eye(Achild[:, 0].size)))
    
    posterior_sigma = np.linalg.inv(precision_parent + N*precision_child)
    posterior_mu = posterior_sigma @ (precision_parent @ Mprior.flatten(order='F')[:, na] + 
                                      precision_child @ Achild.flatten(order='F')[:, na])
    return npr.multivariate_normal(posterior_mu.flatten(), posterior_sigma).reshape(Achild.shape, order='F')



# In[7]:
def leaf_dynamics(Y, X, nu, Lambda, M, V, draw_prior):
    '''
    Obtain sample of leaf dynamics and noise covariance of MNIW conditional posterior
    :param Y: x_{1:T} in list format
    :param X: x_{0:T-1} in list format
    :param nu: prior degree of freedoms
    :param Lambda: prior psd matrix for IW prior
    :param M: prior mean
    :param V: prior column covariance
    :param draw_prior: boolean variable indicating whether we should sample directly from the prior
    :return: sample A, Q
    '''
    if not draw_prior:
        M_posterior, V_posterior, IW_matrix, df_posterior = utils.compute_ss_mniw(X.T, Y.T, nu, Lambda, M, V)
        A, Q = utils.sample_mniw(df_posterior, IW_matrix, M_posterior, V_posterior)
        return A, Q
    else:
        A, Q = utils.sample_mniw(nu, Lambda, M, V)
        return A, Q


# In[8]:
def discrete_latent_recurrent_only(Z, paths, leaf_path, K, X, U, A, Q, R, depth, D_input):
    '''
    Sampling the discrete latent wrt to the leaves of the tree whic is equivalent to sampling a path in the tree. We are
    assuming the recurrence-only framework from Linderman et al. AISTATS 2017.
    :param Z: list of discrete latent used to store the sampled leaf nodes.
    :param paths: list of paths taken used to store the sampled paths.
    :param leaf_path: all possible paths that can be taken in the current tree-structure
    :param K: Number of leaf nodes
    :param X: list of continuous latent states where each entry in the list is another time series
    :param U: list of (deterministic) inputs
    :param A: dynamics of leaf nodes
    :param Q: noise covariances for each leaf node
    :param R: hyperplanes
    :param depth: maximum depth of tree
    :param D_input: dimension of input
    :return: Z, paths with sampled leaf nodes and paths.
    '''
    Qinv = Q + 0
    Qlogdet = np.ones(K)
    for k in range(K):
        Qinv[:, :, k] = np.linalg.inv(Q[:, :, k])
        Qlogdet[k] = np.log(np.linalg.det(2 * np.pi * Q[:, :, k]))
    log_trans = np.ones(K)
    for idx in range(len(X)):
        for t in range(X[idx][0, :].size):

            log_trans = 0 * log_trans
            log_prior_prob = utils.compute_leaf_log_prob(R, X[idx][:, t], K, depth, leaf_path)

            if t != X[idx][0, :].size - 1:
                for k in range(K):
                    "Compute transition probability for each leaf"
                    log_trans[k] = utils.log_mvn(X[idx][:, t + 1], np.matmul(A[:, :-D_input, k], X[idx][:, t]) +
                                           np.matmul(A[:, -D_input:, k], U[idx][:, t]), Q[:, :, k], Qinv[:, :, k],
                                           Qlogdet[k])

            log_p = log_trans + log_prior_prob  # Sum up prior and transition
            post_unnorm = np.exp(log_p - np.max(log_p))
            post_p = post_unnorm / np.sum(post_unnorm)  # Normalize to make a valid density

            choice = npr.multinomial(1, post_p, size=1)
            paths[idx][:, t] = leaf_path[:, np.where(choice[0, :] == 1)[0][0]].ravel()
            Z[idx][t] = np.where(choice[0, :] == 1)[0][0]

    return Z, paths


# In[9]:
def pg_kalman(D_in, D_bias, X, U, P, As, Qs, C, S, Y, paths, Z, omega,
          alphas, Lambdas, R, depth, omegay=None, bern=False):
    '''
    Polya-Gamma augmented kalman filter for sampling the continuous latent states
    :param D_in: dimension of latent space
    :param D_bias: dimension of input
    :param X: list of continuous latnet states. Used to store sampled.
    :param U: list of inputs
    :param P: A 3D array used to store computed covariance matrices
    :param As: Dynamics of leaf nodes
    :param Qs: noise covariance matrices of leaf nodes
    :param C: emission matrix (with affine term appended at the end)
    :param S: emission noise covariance matrix
    :param Y: list of observations of system
    :param paths: paths taken
    :param Z: list of discrete latent states
    :param omega: list of polya-gamma rvs
    :param alphas: used to store prior mean for each time step
    :param Lambdas: used to store prior covaraince for each time step
    :param R: hyperplanes
    :param depth: maximum depth of tree
    :return: sampled continuous latent states stored in X
    '''
    iden = np.eye(D_in)
    
    
    
    "Filter forward"
    for idx in range(len(X)):
        for t in range(X[idx][0, :].size - 1):

            if depth == 1:
                alpha = X[idx][:, t] + 0  # If tree is of depth one then just run kalman filter
                Lambda = P[:, :, t] + 0
            else:
                # Multiply product of PG augmented potentials and the last posterior
                J = 0
                temp_mu = 0
                for d in range(depth - 1):
                    loc = paths[idx][d, t]  # What node did you stop at
                    fin = paths[idx][d + 1, t]  # Where did you go from current node
                    if np.isnan(fin) != True:
                        k = 0.5 * (fin % 2 == 1) - 0.5 * (
                                fin % 2 == 0)  # Did you go left (ODD) or right (EVEN) from current node in tree
                        tempR = np.expand_dims(R[d][:-1, int(loc - 1)], axis=1)
                        J += omega[idx][d, t] * np.matmul(tempR, tempR.T)
                        temp_mu += tempR.T * (k - omega[idx][d, t] * R[d][-1, int(loc - 1)])

                Pinv = np.linalg.inv(P[:, :, t])
                Lambda = np.linalg.inv(Pinv + J)
                alpha = Lambda @ (Pinv @ X[idx][:, t][:, na] + temp_mu.T)
                

            # Store alpha and Lambda for later use
            alphas[:, t] = alpha.flatten() + 0
            Lambdas[:, :, t] = Lambda + 0
            # Prediction
            Q = Qs[:, :, int(Z[idx][t])] + 0
            x_prior = (As[:, :-D_bias, int(Z[idx][t])] @ alpha + \
                        As[:, -D_bias:, int(Z[idx][t])] @ U[idx][:, t][:, na]).flatten()

            P_prior = np.matmul(np.matmul(As[:, :-D_bias, int(Z[idx][t])], Lambda),
                                As[:, :-D_bias, int(Z[idx][t])].T) + Q
            if bern: #If observations are bernoulli
                kt = Y[idx][:, t] - 0.5
                S = np.diag(1 / omegay[idx][:, t])
            
                
            # Compute Kalman gain
            K = np.matmul(P_prior, np.linalg.solve(np.matmul(np.matmul(C[:, :-1], P_prior),
                                                             C[:, :-1].T) + S, C[:, :-1]).T)
            
            # Correction of estimate
            if bern:
                X[idx][:, t + 1] = x_prior + np.matmul(K,  
             kt/omegay[idx][:, t] - np.matmul(C[:, :-1], x_prior) - C[:, -1])
                
            else:
                X[idx][:, t + 1] = x_prior + np.matmul(K, (Y[idx][:, t] - np.matmul(C[:, :-1], x_prior) - C[:, -1]))
            P_temp = np.matmul((iden - np.matmul(K, C[:, :-1])), P_prior)

            P[:, :, t + 1] = np.array((P_temp + P_temp.T) / 2) + 1e-8 * iden  # Numerical stability

        """
        Sample backwards
        """
        # X[idx][:, -1] = np.random.multivariate_normal(np.array(X[idx][:, -1]).ravel(), P[:, :, X[idx][0, :].size - 1])
        X[idx][:, -1] = X[idx][:, -1] + (
                    np.linalg.cholesky(P[:, :, X[idx][0, :].size - 1]) @ npr.normal(size=D_in)[:, na]).ravel()

        for t in range(X[idx][0, :].size - 2, -1, -1):
            # Load in alpha and lambda
            alpha = alphas[:, t]
            Lambda = Lambdas[:, :, t]

            A_tot = As[:, :-D_bias, int(Z[idx][t])]
            B_tot = As[:, -D_bias:, int(Z[idx][t])][:, na]
            Q = Qs[:, :, int(Z[idx][t])]

            Pn = Lambda - np.matmul(Lambda, np.matmul(A_tot.T,
                                                      np.linalg.solve(Q + np.matmul(np.matmul(A_tot, Lambda), A_tot.T),
                                                                      np.matmul(A_tot, Lambda))))
            mu_n = np.matmul(Pn, np.linalg.solve(Lambda, alpha)[:, na] + np.matmul(A_tot.T,
                                                np.linalg.solve(Q, X[idx][:, t + 1][:, na] - np.matmul(B_tot, U[idx][:, t]))))

            # To ensure PSD of matrix
            Pn = 0.5 * (Pn + Pn.T) + 1e-8 * iden

            # Sample
            X[idx][:, t] = (mu_n + np.linalg.cholesky(Pn) @ npr.normal(size=D_in)[:, na]).ravel()

    return X


# In[]:
def chol_pg_kalman(D_in, D_bias, X, U, P, As, Qs, C, S, Y, paths, Z, omega,
              alphas, Lambdas, R, depth, omegay=None, bern=False):
    '''
    Polya-Gamma augmented kalman filter for sampling the continuous latent states
    :param D_in: dimension of latent space
    :param D_bias: dimension of input
    :param X: list of continuous latnet states. Used to store sampled.
    :param U: list of inputs
    :param P: A 3D array used to store computed covariance matrices
    :param As: Dynamics of leaf nodes
    :param Qs: noise covariance matrices of leaf nodes
    :param C: emission matrix (with affine term appended at the end)
    :param S: emission noise covariance matrix
    :param Y: list of observations of system
    :param paths: paths taken
    :param Z: list of discrete latent states
    :param omega: list of polya-gamma rvs
    :param alphas: used to store prior mean for each time step
    :param Lambdas: used to store prior covaraince for each time step
    :param R: hyperplanes
    :param depth: maximum depth of tree
    :return: sampled continuous latent states stored in X
    '''
    iden = np.eye(D_in)

    # Pre-compute the inverse of the noise covariance for sampling backwards
    Qinvs = np.zeros((D_in, D_in, Qs[0, 0, :].size))
    for k in range(Qs[0, 0, :].size):
        temp = np.linalg.inv(np.linalg.cholesky(Qs[:, :, k]))
        Qinvs[:, :, k] = temp.T @ temp

    "Filter forward"
    for idx in range(len(X)):
        for t in range(X[idx][0, :].size - 1):

            if depth == 1:
                alpha = X[idx][:, t] + 0  # If tree is of depth one then just run kalman filter
                Lambda = P[:, :, t] + 0
            else:
                # Multiply product of PG augmented potentials and the last posterior
                J = 0
                temp_mu = 0
                for d in range(depth - 1):
                    loc = paths[idx][d, t]  # What node did you stop at
                    fin = paths[idx][d + 1, t]  # Where did you go from current node
                    if np.isnan(fin) != True:
                        k = 0.5 * (fin % 2 == 1) - 0.5 * (
                                fin % 2 == 0)  # Did you go left (ODD) or right (EVEN) from current node in tree
                        tempR = np.expand_dims(R[d][:-1, int(loc - 1)], axis=1)
                        J += omega[idx][d, t] * np.matmul(tempR, tempR.T)
                        temp_mu += tempR.T * (k - omega[idx][d, t] * R[d][-1, int(loc - 1)])

                "Take cholesky decomposition and then invert that"
                Pinv = np.linalg.inv(P[:, :, t])
                Lambda = np.linalg.inv(Pinv + J)
                alpha = Lambda @ (Pinv @ X[idx][:, t][:, na] + temp_mu.T)

            # Store alpha and Lambda for later use
            alphas[:, t] = alpha.flatten() + 0
            Lambdas[:, :, t] = Lambda + 0
            # Prediction
            Q = Qs[:, :, int(Z[idx][t])] + 0
            x_prior = (As[:, :-D_bias, int(Z[idx][t])] @ alpha + \
                       As[:, -D_bias:, int(Z[idx][t])] @ U[idx][:, t][:, na]).flatten()

            P_prior = np.matmul(np.matmul(As[:, :-D_bias, int(Z[idx][t])], Lambda),
                                As[:, :-D_bias, int(Z[idx][t])].T) + Q
            if bern:  # If observations are bernoulli
                kt = Y[idx][:, t] - 0.5
                S = np.diag(1 / omegay[idx][:, t])

            # Compute Kalman gain
            K = np.matmul(P_prior, np.linalg.solve(np.matmul(np.matmul(C[:, :-1], P_prior),
                                                             C[:, :-1].T) + S, C[:, :-1]).T)

            # Correction of estimate
            if bern:
                X[idx][:, t + 1] = x_prior + np.matmul(K,
                                                       kt / omegay[idx][:, t] - np.matmul(C[:, :-1], x_prior) - C[:,
                                                                                                                -1])

            else:
                X[idx][:, t + 1] = x_prior + np.matmul(K, (Y[idx][:, t] - np.matmul(C[:, :-1], x_prior) - C[:, -1]))
            P_temp = np.matmul((iden - np.matmul(K, C[:, :-1])), P_prior)

            P[:, :, t + 1] = np.array((P_temp + P_temp.T) / 2) + 1e-8 * iden  # Numerical stability

        """
        Sample backwards
        """
        X[idx][:, -1] = X[idx][:, -1] + (
                np.linalg.cholesky(P[:, :, X[idx][0, :].size - 1]) @ npr.normal(size=D_in)[:, na]).ravel()

        for t in range(X[idx][0, :].size - 2, -1, -1):
            # Load in alpha and lambda
            alpha = alphas[:, t]
            Lambda = Lambdas[:, :, t]

            A_tot = As[:, :-D_bias, int(Z[idx][t])]
            B_tot = As[:, -D_bias:, int(Z[idx][t])]
            Qinv = Qinvs[:, :, int(Z[idx][t])]

            Pn = Lambda - Lambda @ A_tot.T @ np.linalg.solve(Q + A_tot @ Lambda @ A_tot.T, A_tot) @ Lambda
            mu_n = Pn @ (np.linalg.solve(Lambda, alpha)[:, na] + A_tot.T @
                         Qinv @ (X[idx][:, t + 1][:, na] - B_tot @ U[idx][:, t][:, na]))

            # To ensure PSD of matrix
            Pn = 0.5 * (Pn + Pn.T) + 1e-8 * iden

            # Sample
            X[idx][:, t] = (mu_n + np.linalg.cholesky(Pn) @ npr.normal(size=D_in)[:, na]).ravel()

    return X

