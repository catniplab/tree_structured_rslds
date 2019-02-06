import numpy as np
import numpy.random as npr
from numpy import newaxis as na
import scipy
from numpy.linalg import LinAlgError
from scipy.stats import invwishart
from pypolyagamma import PyPolyaGamma
import pypolyagamma
from numba import jit
import os
import conditionals
import torch
from torch.autograd import Variable
import torch.optim as optim

# In[1]:
def compute_ss_mniw(X, Y, nu, Lambda, M, V ):
    '''
    Compute sufficient statistics for MNIW posterior of emission parameter
    :param observations: list of numpy array where array are the observations of the underlying time series
    :param states: list of numpy arrays where each array are the continuous latent states
    :param mask: boolean mask used to remove missing data. A list of mask for each time series
    :param nu: prior degree of freedoms
    :param Lambda: prior on noise covariance
    :param M: prior mean of emission
    :param V: prior row covariance
    :return: posterior parameters (M, V, IW, nu)
    '''
    df_posterior = nu + X[:, 0].size #Update degrees of freedom for IW posterior

    Vinv = np.linalg.inv(V) #Precompute to save time
    Ln = X.T @ X + Vinv
    Bn = np.linalg.solve(Ln, X.T @ Y + Vinv @ M.T)
#    IW_matrix = Lambda + (Y - X @ Bn).T @ (Y - X @ Bn) + (Bn - M.T).T @  np.linalg.solve(V, Bn - M.T)
    IW_matrix = Lambda + (Y - X @ Bn).T @ (Y - X @ Bn) + (Bn - M.T).T @  Vinv @ (Bn - M.T)
    # To ensure PSD
    IW_matrix = (IW_matrix + IW_matrix.T) / 2

    M_posterior = Bn.T
    V_posterior = np.linalg.inv(Ln)
    return M_posterior, V_posterior, IW_matrix, df_posterior

# In[2]:
def sample_mniw(nu, L, M, S):
    '''
    Sample from matrix normal inverse wishart distribution defined by the four parameters.
    :param nu: degree of freedom
    :param L: psd matrix for inverse wishart
    :param M: mean
    :param S: row covariance
    :return: (A,Q) from MNIW distribution
    '''
    #Sample from inverse wishart
    Q = invwishart.rvs(nu, L)
    #Sample from Matrix Normal
    A = npr.multivariate_normal(M.flatten(order='F'), np.kron(S, Q)).reshape(M.shape, order='F')
    return A, Q


# In[3]:
def rotate_latent(states, O):
    '''
    Rotate the latent states by the orthogonal matrix O
    :param states: list of continuous latent states
    :param O: orthogonal matrix
    :return: rotated states
    '''
    return [ O @ states[idx] for idx in range(len(states))]


# In[4]:
def rotate_dynamics(A, O, depth):
    '''
    Rotate the dynamics of each node in tree
    :param A: list of array where each array corresponds to the dynamics of a certain level in the tree.
    :param O: orthogonal matrix
    :param depth: maximum depth of tree
    :return: rotated dynamics
    '''
    for level in range(depth):
        for node in range(2 ** level):
            A[level][:, :-1, node] = O @ A[level][:, :-1, node] @ O.T  # Rotate dynamics
            A[level][:, -1, node] = (O @ A[level][:, -1, node][:, na]).ravel()  # Rotate affine term
    return A


# In[5]:
def sample_hyperplanes(states, omega, paths, depth, prior_mu, prior_sigma, possible_paths, R):
    X = np.hstack(states) #stack all continuous latent states
    X = np.vstack((X, np.ones((1, X[0, :].size)))) #append a vector of all ones
    W = np.hstack(omega) #stack all polya-gamma rvs
    path = np.hstack(paths) #append all paths taken through the tree

    for level in range(depth - 1): #Traverse through the tree. Note that only internal nodes have hyperplanes
        for node in range(2 ** level):
            # Check to see if current node is a leaf node or not
            if np.isnan(possible_paths[level + 1, 2 * node + 1]) == False:
                indices = path[level, :] == (node + 1) #Create a boolean mask
                effective_x = X[:, indices]
                effective_w = W[level, indices]
                effective_z = path[level + 1, indices]

                draw_prior = indices.size == 0  # If no data points allocated, draw from the prior

                R[level][:, node] = conditionals.hyper_planes(effective_w, effective_x, effective_z,
                                                              prior_mu, prior_sigma, draw_prior)
    return R


# In[6]:
def sample_internal_dynamics(A, scale, Mx, Vx, depth):
    # Sample from bottoms up
    for level in range(depth - 2, -1, -1):
        for node in range(2 ** level):
            Achild = 0

            # Check to see if current node is a leaf node
            if not np.isnan(A[level + 1][0, 0, 2 * node]):
                if level == 0:  # If root node then prior is Mx
                    Mprior = Mx + 0
                else:
                    Mprior = A[level - 1][:, :, int(np.floor(node/2))] + 0  # If not root then parent is your prior.

                for child in range(2):
                    Achild += A[level + 1][:, :, 2 * node + child]

            A[level][:, :, node] = conditionals._internal_dynamics(Mprior, scale ** level * Vx, Achild,
                                                                   scale ** (level + 1) * Vx)
    return A

# In[7]:
def sample_leaf_dynamics(states, inputs, discrete_states, A, Q, nu, lambdax, Mx, Vx, scale, leaf_nodes):
    X = np.hstack([states[idx][:, :-1] for idx in range(len(states))])  # x_{0:T-1}
    U = np.hstack([inputs[idx][:, :-1] for idx in range(len(inputs))])  # inputs u_{0:T-1}
    X = np.vstack((X, U))

    Y = np.hstack([states[idx][:, 1:] for idx in range(len(states))])  # x_{1:T}

    Z = np.hstack([discrete_states[idx][:-1] for idx in range(len(states))]) #discrete states
    
    for (d, node, k) in leaf_nodes:
        indices = Z == k #create a boolean mask

        effective_X = X[:, indices]
        effective_Y = Y[:, indices]
        if d == 0:
            Mprior = Mx
        else:
            Mprior = A[d-1][:, :, int(np.floor(node/2))]

        draw_prior = effective_X.size == 0  # If no data points allocated, draw from prior.
        A[d][:, :, int(node)], Q[:, :, int(k)] = conditionals.leaf_dynamics(effective_Y, effective_X, nu,
                                                                            lambdax, Mprior, scale ** d * Vx, draw_prior)

    return A, Q



# In[8]:
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

# In[9]:
def log_mvn(x, mu, sigma, tau, logdet):
#    return -0.5*np.log(np.linalg.det(2*np.pi*sigma))-0.5*np.matmul(np.linalg.solve(sigma, x-mu).T, x-mu)
    return -0.5*logdet-0.5 * (x - mu)[na, :] @ tau @ (x - mu)[:, na]

# In[10]:
def compute_leaf_log_prob(R, x, K, depth, leaf_paths):
    # Generate discrete latent states
    log_prob = np.zeros(K)
    for k in range(K):
        "Compute prior probabilities of each path"
        for level in range(depth - 1):
            node = int(leaf_paths[level, k])
            child = int(leaf_paths[level + 1, k])
            v = np.matmul(R[level][:-1, node - 1], x) + R[level][-1, node - 1]
            if child % 2 == 1:  # If odd then you went left
                log_prob[k] += np.log(sigmoid(v))
            else:
                log_prob[k] += np.log(sigmoid(-v))
    return log_prob

# In[11]:
def create_balanced_binary_tree(K):
    depth = int(np.ceil(np.log2(K)) + 1)  # Find the maximum depth of the tree
    # Assume a perfect binary tree
    K_perf = 2 ** (depth - 1)
    possible_paths = np.ones((depth, K_perf))
    for d in range(1, depth):
        temp = np.arange(0, 2 ** int(d)) + 1
        possible_paths[d, :] = np.repeat(temp, int(K_perf / temp.size))

    # Find difference between perfect and balanced binary tree
    right = (2 ** (depth - 1) - K) // 2  # Number of subtrees to remove from right side
    left = (2 ** (depth - 1) - K) - right  # Number of subtrees to remove from left side
    split = K_perf // 2
    possible_paths[-1, :2 * left] = np.nan
    possible_paths[-1, split:split + 2 * right] = np.nan

    # Keep only leaf paths
    leaf_path = np.zeros((depth, K))
    indic = False
    counter = 0
    for n in range(K_perf):
        if n == 0:
            leaf_path[:, counter] = possible_paths[:, n]
            indic = np.isnan(possible_paths[-1, n])
            counter += 1
        elif indic == False:
            leaf_path[:, counter] = possible_paths[:, n]
            indic = np.isnan(possible_paths[-1, n])
            counter += 1
        else:
            indic = False

    leaf_nodes = []  # A list of tuples (d,n,k) where d and n are the depth and node in the tree respectively
    for d in range(depth - 2, depth):  # Check the last two levels
        for k in range(K):
            if d == depth - 1:  # bottom level of tree
                if not np.isnan(leaf_path[d, k]):
                    leaf_nodes.append((int(d), int(leaf_path[d, k] - 1), int(k)))
            else:  # level before bottom level
                if np.isnan(leaf_path[d + 1, k]):
                    leaf_nodes.append((int(d), int(leaf_path[d, k] - 1), int(k)))

    return depth, leaf_path, possible_paths, leaf_nodes

# In[12]:
def optimize_tree(y, x, LDS, nu, ancestor_weights, K, HP, path_LDS, max_epoch, batch_size, LR, temper):
    # dim=dimension of latent space
    # y = output data
    # x = input data
    # LDS= linear dynamics of nodes in current depth of the tree
    # nu= hyperplanes
    # ancestor_weights= prpobability of previous paths
    # K=number of nodes at current depth of the tree
    # HP= number of hypeplanes
    # path_LDS= weighted sum of previous LDS
    # max_epoch = maximum number of epochs
    # batch = size of batch
    # LR = learning rate
    # temper = parameter in sigmoid function used to make decision boundaries sharper

    nT = int(x[:, 0].size)  # Number of trajectories

    N = int(np.ceil(nT / batch_size))
    batch_size = int(batch_size)
    rows, cols = x.T.shape

    input_data = torch.from_numpy(x.T).double()
    output_data = torch.from_numpy(y.T).double()
    LD = Variable(torch.from_numpy(LDS), requires_grad=True).double()
    hp = Variable(torch.from_numpy(nu), requires_grad=True).double()
    prev_weights = torch.from_numpy(ancestor_weights).double()
    p_LDS = Variable(torch.from_numpy(path_LDS)).double()

    # Construct optimizer object
    #    optimizer = optim.SGD( [LD, hp], lr = LR,  momentum=0.95, dampening = 0, nesterov = True )
    optimizer = optim.Adam([LD, hp], lr=LR)
    # Perform optimization
    for epoch in range(max_epoch):
        for n in range(N):
            optimizer.zero_grad()

            if n == N - 1:
                X = Variable(input_data[:, n * batch_size:])
                Y = Variable(output_data[:, n * batch_size:])
                anc_weights = Variable(prev_weights[:, n * batch_size:])
                weights_local = Variable(torch.from_numpy(np.zeros((K, len(input_data[0, n * batch_size:])))))
                y_local = Variable(torch.from_numpy(np.zeros((rows - 1, len(input_data[0, n * batch_size:]), K))))

            else:
                X = Variable(input_data[:, n * batch_size:(n + 1) * batch_size])
                Y = Variable(output_data[:, n * batch_size:(n + 1) * batch_size])
                anc_weights = Variable(prev_weights[:, n * batch_size:(n + 1) * batch_size])
                weights_local = Variable(torch.from_numpy(np.zeros((K, batch_size))))
                y_local = Variable(torch.from_numpy(np.zeros((rows - 1, batch_size, K))))

            # Compute weight of each path
            counter = 0
            for h in range(0, HP):
                weights_local[counter, :] = torch.mul(anc_weights[counter, :],
                                                      torch.sigmoid(temper * torch.matmul(X.transpose(0, 1), hp[:, h])))
                weights_local[counter + 1, :] = torch.mul(anc_weights[counter + 1, :], torch.sigmoid(
                    -temper * torch.matmul(X.transpose(0, 1), hp[:, h])))
                counter += 2

            # Compute weighted sum of LDS
            for k in range(0, K):
                y_local[:, :, k] = torch.mul(weights_local[k, :], torch.matmul(p_LDS[:, :, k] + LD[:, :, k], X))

            y_pred = torch.sum(y_local, 2)

            # Compute difference
            z = Y - y_pred

            # Compute MSE
            loss = torch.matmul(z, z.transpose(0, 1)).trace() / len(X[0, :])

            # Perform backprop
            loss.backward()

            # Update parameters
            optimizer.step()

    return LD, hp
