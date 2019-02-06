#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:09:57 2018

@author: josuenassar
"""

import numpy as np
import numpy.random as npr
import scipy
from numpy.linalg import LinAlgError
from scipy.stats import invwishart
from pypolyagamma import PyPolyaGamma
import pypolyagamma
from numba import jit
import os


# Add a funciton that fixes the random seed for debugging purposes
def debugging():
    np.random.seed(43)

"Sample the dynamic parameters of one node conditioned on all the other ones"
def emission_parameters(obsv, states, nu, Lambda, M, V):
    """
    x = x_[0:T-1]
    y = x_[1:T]
    path = path traversed through the tree
    level= indicating that we are estimating a node in level l - 1
    M, V = parameters for multivariate normal prior
    depth = depth of the tree
    A,B = linear dynamics
    Q = state covraince
    """

    M_posterior, V_posterior, IW_matrix, df_posterior = compute_sufficient_stats_MNIW(obsv, states, nu, Lambda, M, V )

    # Sample S from IW
    S = invwishart.rvs(df_posterior, IW_matrix)

    # Sample from Matrix Normal distribution
    Emission_matrices = MatrixNormal_samples(M_posterior, S, V_posterior)

    C_temp = Emission_matrices[:, :-1]

    #Normalize columns
    L = np.diag( C_temp.T*C_temp )
    L = np.diag( np.power( L, -0.5 ) )
    C_temp=C_temp*L
    Emission_matrices[:, :-1] = C_temp

    return Emission_matrices, S



"For computing ss for MNIW"

def compute_sufficient_stats_MNIW(obsv, states, nu, Lambda, M, V ):
    num_trajec = states[0, 0, :].size
    df_posterior = nu
    for n in range(0, num_trajec):
        x = np.matrix(states[:, :, n])
        y = np.matrix(obsv[:, :, n])
        T = x[0, :].size
        dim = x[:, 0].size
        df_posterior += T
        y = (y.T)
        x = np.matrix(np.append(x, np.ones((1, T)), axis=0)).T
        if n == 0:
            R = y
            X = x
        else:
            R = np.append(R, y, axis=0)
            X = np.append(X, x, axis=0)

    V = np.matrix(V)
    M = np.matrix(M)

    Ln = X.T * X + V.I + 1e-10 * np.eye(dim + 1)
    Bn = np.linalg.solve(Ln, X.T * R)

    df_posterior = nu + num_trajec * y[:, 0].size
    IW_matrix = Lambda + (R - X * Bn).T * (R - X * Bn) + (Bn - M.T).T * np.linalg.solve(V, Bn - M.T)
    # To ensure PSD
    IW_matrix = (IW_matrix + IW_matrix.T) / 2

    M_posterior = Bn.T
    V_posterior = Ln.I
    return M_posterior, V_posterior, IW_matrix, df_posterior



"Sample the dynamic parameters of one node conditioned on all the other ones"
def emission_parameters_annealed(obsv, states, nu, Lambda, M, V, beta):
    """
    x = x_[0:T-1]
    y = x_[1:T]
    path = path traversed through the tree
    level= indicating that we are estimating a node in level l - 1
    M, V = parameters for multivariate normal prior
    depth = depth of the tree
    A,B = linear dynamics
    Q = state covraince
    """

    M_posterior, V_posterior, IW_matrix, df_posterior = compute_sufficient_stats_MNIW_annealed(obsv, states, nu, Lambda, M, V, beta )

    # Sample S from IW
    S = invwishart.rvs(df_posterior, IW_matrix)

    # Sample from Matrix Normal distribution
    Emission_matrices = MatrixNormal_samples(M_posterior, S, V_posterior)

    C_temp = Emission_matrices[:, :-1]

    #Normalize columns
    L = np.diag( C_temp.T*C_temp )
    L = np.diag( np.power( L, -0.5 ) )
    C_temp=C_temp*L
    Emission_matrices[:, :-1] = C_temp

    return Emission_matrices, S



"For computing ss for MNIW"

def compute_sufficient_stats_MNIW_annealed(obsv, states, nu, Lambda, M, V, beta):
    num_trajec = states[0, 0, :].size
    df_posterior = nu
    for n in range(0, num_trajec):
        x = np.sqrt(beta)*np.matrix(states[:, :, n])
        y = np.sqrt(beta)*np.matrix(obsv[:, :, n])
        T = x[0, :].size
        dim = x[:, 0].size
        df_posterior += beta*T
        y = (y.T)
        x = np.matrix(np.append(x, np.ones((1, T)), axis=0)).T
        if n == 0:
            R = y
            X = x
        else:
            R = np.append(R, y, axis=0)
            X = np.append(X, x, axis=0)

    V = np.matrix(V)
    M = np.matrix(M)

    Ln = X.T * X + V.I + 1e-10 * np.eye(dim + 1)
    Bn = np.linalg.solve(Ln, X.T * R)

    df_posterior = nu + num_trajec * y[:, 0].size
    IW_matrix = Lambda + (R - X * Bn).T * (R - X * Bn) + (Bn - M.T).T * np.linalg.solve(V, Bn - M.T)
    # To ensure PSD
    IW_matrix = (IW_matrix + IW_matrix.T) / 2

    M_posterior = Bn.T
    V_posterior = Ln.I
    return M_posterior, V_posterior, IW_matrix, df_posterior


"""
Sample the emission parameters from a Bernoulli GLM
"""
@jit
def emission_parameters_spike_train(Y, X, Wy, mu, Sigma):
    """
    Y = spikes
    X = latent sttaes
    Wy = PG rvs
    mu, Sigma = parameters of prior
    """
    num_trajec = X[0, 0, :].size
    dim_y = Y[:, 0, 0].size
    dim = X[:, 0, 0].size
    C = np.zeros((dim_y, dim + 1))

    for d in range(dim_y):
        Lambda_post = np.linalg.inv(Sigma)
        temp_mu = np.matmul(mu, Lambda_post)

        for n in range(num_trajec):
            x = X[:, :, n]
            y = Y[d, :, n]
            w = Wy[d, :, n]

            # append with 1s and
            x_tilde = np.vstack((x, np.ones(x[0, :].size)))

            # pre multiply by sqrt(w_n,t )
            xw_tilde = np.multiply(x_tilde, np.sqrt(w))

            # Use einstein summation to compute sum of outer products
            Lambda_post += np.einsum('ij,ik->jk', xw_tilde.T, xw_tilde.T)

            # multiply x by k_n,t
            x_tilde[:, y == 1] = 0.5 * x_tilde[:, y == 1]
            x_tilde[:, y == 0] = -0.5 * x_tilde[:, y == 0]

            temp_mu += np.sum(x_tilde.T, axis=0)

        Sigma_post = np.linalg.inv(Lambda_post)
        mu_post = np.matmul(temp_mu, Sigma_post)
        # Sample from mvn posterior
        sample = npr.multivariate_normal(np.array(mu_post).ravel(), Sigma_post)
        C[d, :] = sample

    # Normalize columns of C
    # C_temp = C[:, :dim]
    # L = np.diag(np.matmul(C_temp.T, C_temp))
    # L = np.diag(np.power(L, -0.5))
    # C_temp = np.matmul(C_temp, L)
    # C[:, :dim] = C_temp

    return C


"""
Function that will generate samples from a Matrix Normal distribution parameterized by 
MN(M,U,V)
"""
def MatrixNormal_samples(M, U, V):
    rows = M[:, 0].size
    cols = M[0, :].size

    # Take cholestky decomposition of U and V
    try:
        A = np.linalg.cholesky(U)
    except LinAlgError:
        U += 1e-8 * np.eye(U[:, 0].size)
        A = np.linalg.cholesky(U)

    try:
        B = np.linalg.cholesky(V)
    except LinAlgError:
        V += 1e-8 * np.eye(V[:, 0].size)
        B = np.linalg.cholesky(V)

    # Sample from MN(0,I,I)
    e = np.random.multivariate_normal(np.zeros(rows * cols), np.eye(rows * cols))
    # Reshape into a matrix
    e = np.matrix(np.reshape(e, (rows, cols)))

    # Transform to sample from target matrix normal distribution
    sample = M + A * e * B

    return sample


"""
Sample Polya-Gamma w_n,t|x_t,z_{t+1} where the subscript n denotes the hyperplane 
for which we are augmenting with the Polya-Gamma. Thus will augment all the logistic regressions 
that was taken while traversing down the tree
"""
#@jit
def PG_tree_posterior(states, w, R, r, path, depth):
    """
    states: This variable contains the continuous latent states
    R,r: parameters that characterize the hyper-plane
    path: path taken through the tree at time t
    depth: depth of the tree
    """
    T = states[0, :].size  # find length of time series
    b = np.ones(T)
    nthreads = T
    v = np.ones(T)
    out = np.empty(T)
    for d in range(0, depth - 1):    
        for t in range(0, T):
            index = int(path[d, t] - 1)  # Find which node you went through
            v[t] = np.matmul(R[d][:,index], np.array(states[:,t]) ) + r[d][index]
#            w[d, t] = pg.pgdraw(1, v)  # Sample from polya gamma
        seeds = np.random.randint(2**16, size=nthreads)
        ppgs = [PyPolyaGamma(seed) for seed in seeds]
        
        # Sample in parallel
        pypolyagamma.pgdrawvpar(ppgs, b, v, out)
        w[d, :] = out
    
    
    return w


"""
Sample Polya-Gamma wy|Y,C,D,X where Y are spike trains and X are the continuous latent states
"""
#@jit
def PG_spike_train(Y, X, C, D):
    """
    Y = spike trains
    X = continuous latent states from (1,T) i.e. ignoring starting point
    C, D = emission parameters
    """
    T = X[0, :].size  # Length of time series
    b = np.ones(T)
    nthreads = T
    out = np.empty(T)
    dim_y = Y[:, 0].size
    w = np.zeros((dim_y, T))
    C = np.array(C)
    D = np.expand_dims(np.array(D).flatten(), axis = 1)
    V = np.matmul(C, np.array(X)) + np.array(D)
#    for t in range(T):
#        for d in range(dim_y):
#            w[d, t] = pg.pgdraw(1, V[d, t])
    for level in range(dim_y):
        seeds = np.random.randint(2**16, size=nthreads)
        ppgs = [PyPolyaGamma(seed) for seed in seeds]
        
        # Sample in parallel
        pypolyagamma.pgdrawvpar(ppgs, b, V[level, :], out)
        w[level, :] = out

    return w

"""
Sample the hyper-planes
"""
def hyper_planes(pg_rv, states, discrete_states, mu_prior, Sigma_prior, temper, draw_prior, dim):
    """
    pg_rv = PG random variables
    states= list of continuous latent states from different trajectories
    discrete_states= list of discrete latent states
    mu_prior, Sigma_prior= hyper parameters of prior on hyper planes
    temper = parameter in sigmoid function used to make the decision boundaries sharper
    draw_prior = If there are no data points then we will draw from the prior instead
    """
    num_trajec = len(states)

    mu_prior = np.matrix(mu_prior)
    Sigma_prior = np.matrix(Sigma_prior)

    Precision_post = Sigma_prior.I
    mu_inter = np.linalg.solve(Sigma_prior, mu_prior)

    if draw_prior == False:
        for n in range(0, num_trajec):
            x = np.matrix(states[n])
            T = x[0, :].size
            w = np.array(pg_rv[n])
            d_states = np.array(discrete_states[n])
            for t in range(0, T):
                u = temper * np.append(x[:, t], [[1]], axis=0)
                k = 0.5 * (d_states[t] % 2 == 1) - 0.5 * (d_states[t] % 2 == 0)
                Precision_post += u * u.T * w[t]
                mu_inter += k * u
        Sigma_post = Precision_post.I
        mu_post = np.linalg.solve(Precision_post, mu_inter)
        Gamma = np.random.multivariate_normal(np.array(mu_post).ravel(), Sigma_post)
    else:
        Gamma = np.random.multivariate_normal(np.array(mu_prior).ravel(), Sigma_prior)
    return Gamma[0:dim], Gamma[dim]


"MNIW posterior for the dynamics of the leaf nodes"
def leaf_dynamics( states, inputs, dim_input, data,  M, V, nu, Lambda, draw_prior ):
    """
    :param x = x_[0:T-1]
    :param y = x_[1:T]
    :param path = path traversed through the tree
    :param level = indicating that we are estimating a node in level l - 1
    :param M, V = parameters for multivariate normal prior
    :param depth = depth of the tree
    :param A,B = linear dynamics
    :param Q = state covraince
    :param draw_prior: boolean variable that indicates whether you should sample from the prior or not
    """
    if not draw_prior :
        M_posterior, V_posterior, IW_matrix, df_posterior = compute_leaf_ss(states, inputs, dim_input, data,  M, V, nu, Lambda)
        # Sample Q from IW
        Q = invwishart.rvs(df_posterior, IW_matrix)

        # Sample from Matrix Normal distribution
        temp = MatrixNormal_samples(M_posterior, Q, V_posterior)
        A = temp[:, :-dim_input]
        B = temp[:, -dim_input:]
    else:
        V = np.matrix(V)
        M = np.matrix(M)
        # Sample Q from IW
        Q = invwishart.rvs(nu, Lambda)

        # Sample from Matrix Normal distribution
        temp = MatrixNormal_samples(M, Q, V)
        A = temp[:, :-dim_input]
        B = temp[:, -dim_input:]

    return A, B, Q



"Helper function to speed up computation of posterior"
def compute_leaf_ss(states, inputs, dim_input, data,  M, V, nu, Lambda ):
    num_trajec = len(states)
    df_posterior = nu

    dim = states[0][:, 0].size
    for n in range(num_trajec):
        x = np.matrix(states[n])
        y = np.matrix(data[n])
        u = np.matrix(inputs[n])
        df_posterior += x[0, :].size
        r = y.T
        x = np.matrix(np.append(x, u, axis=0)).T

        if n == 0:
            R = r
            X = x
        else:
            R = np.append(R, r, axis=0)
            X = np.append(X, x, axis=0)

    V = np.matrix(V)
    M = np.matrix(M)

    Ln = X.T * X + V.I + 1e-10 * np.eye(dim + dim_input)
    Bn = np.linalg.solve(Ln, X.T * R + V.I * M.T)

    IW_matrix = Lambda + (R - X * Bn).T * (R - X * Bn) + (Bn - M.T).T * np.linalg.solve(V, Bn - M.T)
    # To ensure PSD
    IW_matrix = (IW_matrix + IW_matrix.T) / 2
    M_posterior = Bn.T
    V_posterior = Ln.I

    return M_posterior, V_posterior, IW_matrix, df_posterior


"MNIW posterior for the dynamics of the leaf nodes"
def leaf_dynamics_annealed( states, inputs, dim_input, data, 
                           M, V, nu, Lambda, draw_prior, beta ):
    """
    :param x = x_[0:T-1]
    :param y = x_[1:T]
    :param path = path traversed through the tree
    :param level = indicating that we are estimating a node in level l - 1
    :param M, V = parameters for multivariate normal prior
    :param depth = depth of the tree
    :param A,B = linear dynamics
    :param Q = state covraince
    :param draw_prior: boolean variable that indicates whether you should sample from the prior or not
    """
    if not draw_prior :
        M_posterior, V_posterior, IW_matrix, df_posterior = compute_leaf_ss_annealed(states, inputs, dim_input, data,  M, V, nu, Lambda, beta)
        # Sample Q from IW
        Q = invwishart.rvs(df_posterior, IW_matrix)

        # Sample from Matrix Normal distribution
        temp = MatrixNormal_samples(M_posterior, Q, V_posterior)
        A = temp[:, :-dim_input]
        B = temp[:, -dim_input:]
    else:
        V = np.matrix(V)
        M = np.matrix(M)
        # Sample Q from IW
        Q = invwishart.rvs(nu, Lambda)

        # Sample from Matrix Normal distribution
        temp = MatrixNormal_samples(M, Q, V)
        A = temp[:, :-dim_input]
        B = temp[:, -dim_input:]

    return A, B, Q



"Helper function to speed up computation of posterior"
def compute_leaf_ss_annealed(states, inputs, dim_input, data,  M, V, nu, Lambda, beta ):
    num_trajec = len(states)
    df_posterior = nu

    dim = states[0][:, 0].size
    for n in range(num_trajec):
        x = np.sqrt(beta) * np.matrix(states[n])
        y = np.sqrt(beta) * np.matrix(data[n])
        u = np.sqrt(beta) * np.matrix(inputs[n])
        df_posterior += beta*x[0, :].size
        r = y.T
        x = np.matrix(np.append(x, u, axis=0)).T

        if n == 0:
            R = r
            X = x
        else:
            R = np.append(R, r, axis=0)
            X = np.append(X, x, axis=0)

    V = np.matrix(V)
    M = np.matrix(M)

    Ln = X.T * X + V.I + 1e-10 * np.eye(dim + dim_input)
    Bn = np.linalg.solve(Ln, X.T * R + V.I * M.T)

    IW_matrix = Lambda + (R - X * Bn).T * (R - X * Bn) + (Bn - M.T).T * np.linalg.solve(V, Bn - M.T)
    # To ensure PSD
    IW_matrix = (IW_matrix + IW_matrix.T) / 2
    M_posterior = Bn.T
    V_posterior = Ln.I

    return M_posterior, V_posterior, IW_matrix, df_posterior


"Posteriior of isotropic prior of node dynamics"
def compute_prior_covariance(child, parent):
    n = child.size #Number of data points: n*(n+1)
    beta = np.sum(np.power(child-parent, 2))
    tau = scipy.stats.invgamma.rvs(a= 1 + n/2, scale=1 + beta/2)
    return tau

def interior_dynamics_pt2(child1, tau1, child2, tau2, prior, tau_prior, dim_input):
    """
    :param child1: Dynamics of child1
    :param tau1: prior covariance is isotropic = tau1*I
    :param child2: Dynamics of child2
    :param tau2: prior covariance is isotropic = tau2*I
    :param prior: Prior on current interior nodes dynamcis
    :param tau_prior: Prior covariance is isotropic = tau_prior*I
    :param dim_input: dimension of input
    :return: Sample from conditional posterior
    """
    rows = child1[:, 0].size
    cols = child1[0, :].size
    
    taun = 1/(1/tau1 + 1/tau2 + 1/tau_prior)
    Bn = taun*(child1/tau1 + child2/tau2 + prior/tau_prior)

    #Obtain sample from Matrix Normal posterior
    temp = MatrixNormal_samples(Bn, taun * np.eye(rows), np.eye(cols))
    An = temp[:, :-dim_input]
    Bn = temp[:, -dim_input:]

    return An, Bn

"""
Matrix Normal posterior for dynamics of interior nodes
"""
def interior_dynamics(A, B, U, M, V, dim_input):
    """
    :param A: sum of children LDS
    :param B: sum of children affine term
    :param U: row covariance of likelihood
    :param M: prior mean
    :param V: prior row covariance
    :return: An, Bn sample from MN posterior
    """

    Ac = np.matrix(np.hstack((A, np.matrix(B))))
    U = np.matrix(U)
    M = np.matrix(M)
    V = np.matrix(V)


    #Compute parameters of posterior
    Vn = (2*U.I + V.I).I
    Bn = Vn*(np.linalg.solve(U, Ac) + np.linalg.solve(V, M))

    #Obtain sample from Matrix Normal posterior
    temp = MatrixNormal_samples(Bn, Vn, np.eye(Ac[0, :].size))

    An = temp[:, :-dim_input]
    Bn = temp[:, -dim_input:]

    return An, Bn

"""
Polya-Gamma Augmented  Kalman Filter
"""
@jit
def PG_KF(dim, x, u, P, As, Bs, Qs, C, D, S, T, y, path, z, w,
          alphas, Lambdas, R, r, depth):
    """
    dim=dimension of continuous latent space
    x = continuous latent states so don't have to waste time allocating new memory
    u = input
    P = used to hold posterior covariances
    As, Bs= effective linear dynamics of the tree
    Qs= Covariance of state noise
    C, D, S= Emission parameters
    T= Length of time series
    y= data
    path= Path taken at time t
    z = leaf node selected at time t
    w= PG random variables associated with the path at time t
    alphas = place holder for prior means
    R,r= hyper planes'
    depth= depth of tree
    """
    iden = np.eye(dim)
    C = np.array(C)
    D = np.array(D).flatten()
    """
    Filter forward
    """
    for t in range(0, T):

        if depth == 1:
            alpha = x[:, t]
            Lambda = P[:, :, t]
        else:
            # Multiply product of PG augmented potentials and the last posterior
            J = 0
            temp_mu = 0
            for d in range(0, depth - 1):
                loc = path[d, t]  # What node did you stop at
                fin = path[d + 1, t]  # Where did you go from current node
                if np.isnan(fin) != True:
                    k = 0.5 * (fin % 2 == 1) - 0.5 * (
                                fin % 2 == 0)  # Did you go left (ODD) or right (EVEN) from current node in tree
                    tempR = np.expand_dims( R[d][:, int(loc - 1)] , axis = 1)
                    J += w[d, t] * np.matmul(tempR, tempR.T)
                    temp_mu += tempR.T* (k - w[d, t] * r[d][int(loc - 1)]) 

            Lambda = np.linalg.inv(np.linalg.inv(P[:, :, t]) +J)
            alpha = np.matmul(Lambda, (np.linalg.solve(P[:, :, t], x[:, t]) + temp_mu.flatten()) )
        
        #Store alpha and Lambda for later use
        alphas[:,t] = alpha
        Lambdas[:,:,t] = Lambda
        # Prediction
        Q = Qs[:, :, int(z[t])]
        x_prior = np.matmul(As[:, :, int(z[t])], alpha) + np.matmul(Bs[:, :, int(z[t])], u[:, t])
        P_prior = np.matmul(np.matmul(As[:, :, int(z[t])], Lambda), As[:, :, int(z[t])].T) + Q
        # Compute Kalman gain
        K = np.matmul(P_prior,  np.linalg.solve(np.matmul(np.matmul(C, P_prior), C.T) + S, C).T)
        # Correction of estimate
        x[:, t + 1] = x_prior + np.matmul(K,  (y[:, t] - np.matmul(C, x_prior) - D))
        P_temp = np.matmul((iden - np.matmul(K, C)), P_prior)

        P[:, :, t + 1] = np.array((P_temp + P_temp.T) / 2) + 1e-8 * iden

    """
    Sample backwards
    """
    x[:, T] = np.random.multivariate_normal(np.array(x[:, T]).ravel(), P[:, :, T])

    for t in range(T - 1, -1, -1):
        #Load in alpha and lambda
        alpha = alphas[:, t]
        Lambda = Lambdas[:, :, t]
        
        A_tot = As[:, :, int(z[t])]
        B_tot = Bs[:, :, int(z[t])]
        Q = Qs[:, :, int(z[t])] 

        Pn = Lambda - np.matmul(Lambda, np.matmul(A_tot.T, np.linalg.solve(Q + np.matmul(np.matmul(A_tot,Lambda), A_tot.T), np.matmul(A_tot, Lambda))))
        mu_n = np.matmul(Pn, np.linalg.solve(Lambda, alpha) + np.matmul(A_tot.T, np.linalg.solve(Q, x[:, t+1] - np.matmul(B_tot, u[:, t]))))
        
        # To ensure PSD of matrix
        Pn = 0.5 * (Pn + Pn.T) + 1e-8 * iden

        # Sample
        x[:, t] = np.random.multivariate_normal(np.array(mu_n).ravel(), Pn)

    return x


"""
Polya-Gamma Augmented  Kalman Filter for bernoulli observations
"""
@jit
def PG_KF_spike(dim, x, u, P, As, Bs, Qs, C, D, T, y, path, z, w, wy,
          alphas, Lambdas, R, r, depth):
    """
    dim=dimension of continuous latent space
    As, Bs= effective linear dynamics of the tree
    Qs= Covariance of state noise
    C, D, S= Emission parameters
    T= Length of time series
    x_init, P_init= initial continuous state and covariance matrix
    y= data
    path= Path taken at time t
    z = leaf node selected at time t
    w= PG random variables associated with the path at time t
    Rs,rs= hyper planes'
    depth= depth of tree
    """
    iden = np.eye(dim)
    C= np.array(C)
    D = np.array(D).flatten()

    "Filter forward"
    for t in range(0, T):

        if depth == 1:
            alpha = x[:, t]
            Lambda = P[:, :, t]
        else:
            # Multiply product of PG augmented potentials and the last posterior
            J = 0
            temp_mu = 0
            for d in range(0, depth - 1):
                loc = path[d, t]  # What node did you stop at
                fin = path[d + 1, t]  # Where did you go from current node
                if np.isnan(fin) != True:
                    k = 0.5 * (fin % 2 == 1) - 0.5 * (
                                fin % 2 == 0)  # Did you go left (ODD) or right (EVEN) from current node in tree
                    tempR = np.expand_dims(R[d][:, int(loc - 1)], axis=1)
                    J += w[d, t] * np.matmul(tempR, tempR.T)
                    temp_mu += tempR.T * (k - w[d, t] * r[d][int(loc - 1)])

            Lambda = np.linalg.inv(np.linalg.inv(P[:, :, t]) + J)
            alpha = np.matmul(Lambda, (np.linalg.solve(P[:, :, t], x[:, t]) + temp_mu.flatten()))
        
        #Store alpha and Lambda for later use
        alphas[:,t] = alpha
        Lambdas[:,:,t] = Lambda
        # Prediction
        Q = Qs[:, :, int(z[t])]
        x_prior = np.matmul(As[:, :, int(z[t])], alpha) + np.matmul(Bs[:, :, int(z[t])], u[:, t])
        P_prior = np.matmul(np.matmul(As[:, :, int(z[t])], Lambda), As[:, :, int(z[t])].T) + Q

        kt = y[:, t] - 0.5
        W = np.diag(wy[:,t])
        S = np.linalg.inv(W)
        
        # Compute Kalman gain
        K = np.matmul(P_prior,  np.linalg.solve(np.matmul(np.matmul(C, P_prior), C.T) + S, C).T)
        # Correction of estimate
        x[:, t + 1] = x_prior + np.matmul(K,  (np.linalg.solve(W, kt) - np.matmul(C, x_prior) - D))
        P_temp = np.matmul((iden - np.matmul(K, C)), P_prior)
        P[:, :, t + 1] = np.array((P_temp + P_temp.T) / 2) + 1e-8 * iden


    "Sample backwards"
    x[:, T] = np.random.multivariate_normal(np.array(x[:, T]).ravel(), P[:, :, T])

    for t in range(T - 1, -1, -1):
        #Load in alpha and lambda
        alpha = alphas[:, t]
        Lambda = Lambdas[:, :, t]
        
        A_tot = As[:, :, int(z[t])]
        B_tot = Bs[:, :, int(z[t])]
        Q = Qs[:, :, int(z[t])] 

        Pn = Lambda-np.matmul(Lambda, np.matmul(A_tot.T, np.linalg.solve(Q + np.matmul(np.matmul(A_tot, Lambda), A_tot.T), np.matmul(A_tot, Lambda))))
        mu_n = np.matmul(Pn, np.linalg.solve(Lambda, alpha) + np.matmul(A_tot.T, np.linalg.solve(Q, x[:, t+1] - np.matmul(B_tot, u[:, t]))))
        
        # To ensure PSD of matrix
        Pn = 0.5 * (Pn + Pn.T) + 1e-8 * iden

        # Sample
        x[:, t] = np.random.multivariate_normal(np.array(mu_n).ravel(), Pn)

    return x


"""
Sampling the discrete latent states wrt to leaves of the tree
which is equivalent to sampling a path in the tree
"""
@jit
def discrete_latent(z, path, T, leaf_path, K, x, u, eff_A, eff_B, Q, R, r, depth):
    """
    leaf_path = the possible unique paths in the tree
    K = number of leaf nodes
    states= the continuous latent states
    eff_A, eff_ B= effective dynamics of the system
    Q= covariance of state noise
    R,r= parameters of the hyper plane
    depth = depth of the tree
    beta = temperature parameter
    """

    for t in range(T):
        """
        Compute prior probabilites of each path
        """
        log_prior_prob = np.zeros((K, 1))
        for k in range(K):
            for d in range(depth - 1):
                idx = int(leaf_path[d, k])
                child_idx = leaf_path[d + 1, k]
                if np.isnan(child_idx) == False:
                    child_idx = int(child_idx)
                    v = np.matmul(R[d][:, idx - 1], x[:, t]) + r[d][idx - 1]
                    # If odd then you went left
                    if child_idx % 2 == 1:
                        log_prior_prob[k] = log_prior_prob[k] + np.log(sigmoid(v))
                    else:
                        log_prior_prob[k] = log_prior_prob[k] + np.log(sigmoid(-v))

        if t != T - 1:
            """
            Compute transition probability for each path
            """
            log_trans = np.ones((K, 1))
            for k in range(0, K):
                log_trans_temp = log_mvn(x[:, t+1], np.matmul(eff_A[:, :, k], x[:, t])+np.matmul(eff_B[:, :, k], u[:, t]), Q[:, :, k])
                log_trans[k] = log_trans_temp

            # Multiply prior and transition probability
            log_p_post = log_trans + log_prior_prob
        else:
            log_p_post = log_prior_prob

        # Normalize to make it a valid density
        p_unnorm = np.exp(log_p_post - np.max(log_p_post))
        p_norm = p_unnorm / np.sum(p_unnorm)


        # Sample from the possibilities of paths
        choice = np.random.multinomial(1, p_norm.ravel(), size=1)
        path[:, t] = leaf_path[:, np.where(choice[0, :] == 1)].ravel()
        z[t] = np.where(choice[0, :] == 1)[0][0]

    return z, path

@jit
def discrete_latent_anneal(z, path, T, leaf_path, K, x, u, eff_A, eff_B, Q, R, r, depth, beta):
    """
    leaf_path = the possible unique paths in the tree
    K = number of leaf nodes
    states= the continuous latent states
    eff_A, eff_ B= effective dynamics of the system
    Q= covariance of state noise
    R,r= parameters of the hyper plane
    depth = depth of the tree
    beta = temperature parameter
    """

    for t in range(T):
        """
        Compute prior probabilites of each path
        """
        log_prior_prob = np.zeros((K, 1))
        for k in range(K):
            for d in range(depth - 1):
                idx = int(leaf_path[d, k])
                child_idx = leaf_path[d + 1, k]
                if np.isnan(child_idx) == False:
                    child_idx = int(child_idx)
                    v = np.matmul(R[d][:, idx - 1], x[:, t]) + r[d][idx - 1]
                    # If odd then you went left
                    if child_idx % 2 == 1:
                        log_prior_prob[k] = log_prior_prob[k] + np.log(sigmoid(v))
                    else:
                        log_prior_prob[k] = log_prior_prob[k] + np.log(sigmoid(-v))

        if t != T - 1:
            """
            Compute transition probability for each path
            """
            log_trans = np.ones((K, 1))
            for k in range(0, K):
                log_trans_temp = log_mvn(x[:, t+1], np.matmul(eff_A[:, :, k], x[:, t])+np.matmul(eff_B[:, :, k], u[:, t]), Q[:, :, k])
                log_trans[k] = log_trans_temp

            # Multiply prior and transition probability
            log_p_post = beta*log_trans + log_prior_prob
        else:
            log_p_post = log_prior_prob

        # Normalize to make it a valid density
        p_unnorm = np.exp(log_p_post - np.max(log_p_post))
        p_norm = p_unnorm / np.sum(p_unnorm)


        # Sample from the possibilities of paths
        choice = np.random.multinomial(1, p_norm.ravel(), size=1)
        path[:, t] = leaf_path[:, np.where(choice[0, :] == 1)].ravel()
        z[t] = np.where(choice[0, :] == 1)[0][0]

    return z, path



@jit
def log_mvn(x, mu, sigma):
    return -0.5*np.log(np.linalg.det(2*np.pi*sigma))-0.5*np.matmul(np.linalg.solve(sigma, x-mu).T, x-mu)


@jit
def sigmoid(x):
    "Numerically stable sigmoid function."
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1 + z)
