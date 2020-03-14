import numpy as np
import numpy.random as npr
from numpy import newaxis as na
from scipy.stats import invwishart
from . import conditionals
import torch
from torch.autograd import Variable
import torch.optim as optim
import copy
from tqdm import tqdm
from scipy.ndimage import filters
from scipy.signal import gaussian
from numba import njit, jit


# In[1]:
def compute_ss_mniw(X, Y, nu, Lambda, M, V ):
    """
    Compute sufficient statistics for MNIW posterior of emission parameter
    :param X: Inputs for regression
    :param Y:  Target data
    :param nu: prior degrees of freedom
    :param Lambda: Prior psd matrix for inverse wishart
    :param M:  Prior mean for matrix normal
    :param V: Prior column covariance matrix
    :return: Parameters of posterior (M, V, IW, nu)
    """
    df_posterior = nu + X[:, 0].size  # Update degrees of freedom for IW posterior

    Vinv = np.linalg.inv(V)  # Precompute to save time
    Ln = X.T @ X + Vinv
    Bn = np.linalg.solve(Ln, X.T @ Y + Vinv @ M.T)
    IW_matrix = Lambda + (Y - X @ Bn).T @ (Y - X @ Bn) + (Bn - M.T).T @  Vinv @ (Bn - M.T)
    # To ensure PSD
    IW_matrix = (IW_matrix + IW_matrix.T) / 2

    M_posterior = Bn.T
    V_posterior = np.linalg.inv(Ln)
    return M_posterior, V_posterior, IW_matrix, df_posterior


# In[2]:
def sample_mniw(nu, L, M, S):
    """
    Sample from matrix normal inverse wishart distribution defined by the four parameters.
    :param nu: degree of freedom
    :param L: psd matrix for inverse wishart
    :param M: mean
    :param S: row covariance
    :return: (A,Q) from MNIW distribution
    """
    # Sample from inverse wishart
    Q = invwishart.rvs(nu, L)
    # Sample from Matrix Normal
    A = npr.multivariate_normal(M.flatten(order='F'), np.kron(S, Q)).reshape(M.shape, order='F')
    return A, Q


# In[3]:
def rotate_latent(states, O):
    """
    Rotate the latent states by the orthogonal matrix O
    :param states: list of continuous latent states
    :param O: orthogonal matrix
    :return: rotated states
    """
    return [O @ states[idx] for idx in range(len(states))]


# In[4]:
def rotate_dynamics(A, O, depth):
    """
    Rotate the dynamics of each node in tree
    :param A: list of array where each array corresponds to the dynamics of a certain level in the tree.
    :param O: orthogonal matrix
    :param depth: maximum depth of tree
    :return: rotated dynamics
    """
    for level in range(depth):
        for node in range(2 ** level):
            A[level][:, :-1, node] = O @ A[level][:, :-1, node] @ O.T  # Rotate dynamics
            A[level][:, -1, node] = (O @ A[level][:, -1, node][:, na]).ravel()  # Rotate affine term
    return A


# In[5]:
def sample_hyperplanes(states, omega, paths, depth, prior_mu, prior_tau, possible_paths, R):
    """
    Sample from conditional posterior of hyperplanes
    :param states: continuous latent states
    :param omega: augmented polya-gamma random variables
    :param paths: path that each latent state took
    :param depth: depth of tree
    :param prior_mu: prior mean
    :param prior_tau: prior covariance
    :param possible_paths: possible paths that can be taken in the tree
    :param R: Used to hold samples
    :return: R
    """
    X = np.hstack(states)  # stack all continuous latent states
    X = np.vstack((X, np.ones((1, X[0, :].size))))  # append a vector of all ones
    W = np.hstack(omega)  # stack all polya-gamma rvs
    path = np.hstack(paths)  # append all paths taken through the tree

    for level in range(depth - 1):  # Traverse through the tree. Note that only internal nodes have hyperplanes
        for node in range(2 ** level):
            # Check to see if current node is a leaf node or not
            if ~np.isnan(possible_paths[level + 1, 2 * node + 1]):
                indices = path[level, :] == (node + 1)  # Create a boolean mask
                effective_x = X[:, indices]
                effective_w = W[level, indices]
                effective_z = path[level + 1, indices]

                draw_prior = indices.size == 0  # If no data points allocated, draw from the prior

                R[level][:, node] = conditionals.hyper_planes(effective_w, effective_x, effective_z,
                                                              prior_mu, prior_tau, draw_prior)
    return R


# In[6]:
def sample_internal_dynamics(A, scale, Mx, Vx, depth):
    """
    Sample from conditional posterior of the internal dynamics
    :param A: Used to store samples
    :param scale: Scale parameter that controls the closeness between parent and child
    :param Mx: Prior Mean
    :param Vx: Prior Row Covariance
    :param depth: maximum depth of tree
    :return: A
    """
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
    """
    Sample from posterior of the dynamics at the leaf nodes
    :param states: continuous latent states
    :param inputs: inputs to the model
    :param discrete_states: discrete latent states
    :param A: Used to store samples
    :param Q: Used to store samples for the noise covariance matrix
    :param nu: prior df for IW
    :param lambdax: prior PSD matrix for IW
    :param Mx:  prior mean for matrix normal
    :param Vx: prior column covraince matrix
    :param scale: parameter that controls the closeness between parents and children
    :param leaf_nodes: location of leaf nodes in the tree
    :return: A, Q
    """
    X = np.hstack([states[idx][:, :-1] for idx in range(len(states))])  # x_{0:T-1}
    U = np.hstack([inputs[idx][:, :-1] for idx in range(len(inputs))])  # inputs u_{0:T-1}
    X = np.vstack((X, U))

    Y = np.hstack([states[idx][:, 1:] for idx in range(len(states))])  # x_{1:T}

    Z = np.hstack([discrete_states[idx][:-1] for idx in range(len(states))])  # discrete states
    
    for (d, node, k) in leaf_nodes:
        indices = Z == k  # create a boolean mask

        effective_X = X[:, indices]
        effective_Y = Y[:, indices]
        if d == 0:
            Mprior = Mx
        else:
            Mprior = A[d-1][:, :, int(np.floor(node/2))]

        draw_prior = effective_X.size == 0  # If no data points allocated, draw from prior.
        A[d][:, :, int(node)], Q[:, :, int(k)] = conditionals.leaf_dynamics(effective_Y, effective_X, nu, lambdax,
                                                                            Mprior, scale ** d * Vx, draw_prior)
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


# In[]
def sigmoid_vectorized(x):
    """
    Vectorized numerically stable sigmoid function
    :param x: input values
    :return: z
    """
    z = np.zeros(x.size)
    z[x >= 0] = 1 / (1 + np.exp(-x[x >= 0]))
    z[x < 0] = np.exp(x[x < 0]) / (1 + np.exp(x[x < 0]))
    return z


# In[9]:
def log_mvn(x, mu, tau, logdet):
    """
    Easy way to compute log multivariate normal with same covariance but different means
    :param x: data, D (dimension) by N (data points)
    :param mu: means, D  by N
    :param tau: D by D inverse covariance
    :param logdet: logdet of tau
    :return:
    """
    return np.diag(-0.5 * logdet-0.5 * (x - mu).T @ tau @ (x - mu))


# In[10]:
def compute_leaf_log_prob(R, x, K, depth, leaf_paths):
    """
    Compute the pmf of discrete latent states according to the tree
    :param R: list of hyperplanes where R[n] are the hyperplanes for the nth level in the tree
    :param x: continuous latent state
    :param K: Number of leaf nodes
    :param depth: depth of the tree
    :param leaf_paths: paths that lead to leaf nodes
    :return: pmf over all possible K leaf nodes
    """
    # Generate discrete latent states
    log_prob = np.zeros(K)
    for k in range(K):
        "Compute prior probabilities of each path"
        for level in range(depth - 1):
            node = int(leaf_paths[level, k])
            child = leaf_paths[level + 1, k]
            if ~np.isnan(child):
                v = np.matmul(R[level][:-1, node - 1], x) + R[level][-1, node - 1]
                if int(child) % 2 == 1:  # If odd then you went left
                    log_prob[k] += np.log(sigmoid(v))
                else:
                    log_prob[k] += np.log(sigmoid(-v))
    return log_prob


# In[10]:
def compute_leaf_log_prob_vectorized(R, x, K, depth, leaf_paths):
    """
    Vectorized version where the pmf over leaf nodes are computed over multiple query points
    :param R: list of hyperplanes where R[n] are the hyperplanes for the nth level in the tree
    :param x: A dx by N array where each column is a query point
    :param K: Number of leaf nodes
    :param depth: depth of the tree
    :param leaf_paths: paths that lead to leaf nodes
    :return: A K by N matrix where each column is the corresponding pmf over the leaf nodes for nth query
    """
    # Generate discrete latent states
    log_prob = np.zeros((K, x[0, :].size))
    for k in range(K):
        "Compute prior probabilities of each path"
        for level in range(depth - 1):
            node = int(leaf_paths[level, k])
            child = leaf_paths[level + 1, k]
            if ~np.isnan(child):
                v = (R[level][:-1, node - 1][na, :] @ x).flatten() + R[level][-1, node - 1]
                if int(child) % 2 == 1:  # If odd then you went left
                    log_prob[k, :] = log_prob[k, :] + np.log(sigmoid_vectorized(v))
                else:
                    log_prob[k, :] = log_prob[k, :] + np.log(sigmoid_vectorized(-v))
    return log_prob


# In[11]:
def create_balanced_binary_tree(K):
    """
    Create a balanced binary tree for a specified number of leaf nodes
    :param K: number of leaf nodes
    :return: depth of the tree, paths for the leaf nodes, all possible paths, location of leaf nodes
    """
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


# In[]
def create_batches(batch_size, N):
    """
    Will create an iterable that will partition data into random batches for SGD
    :param batch_size: number of elements in a batch
    :param N: number of total samples
    :return: a list of indices
    """
    if batch_size == N:
        idx = [np.arange(N)]
    else:
        indices = npr.choice(N, N).astype('int')
        numBatches = np.ceil(N / batch_size).astype('int')
        idx = [indices[i * batch_size: (i + 1) * batch_size] for i in range(numBatches)]
    return idx


# In[12]:
def optimize_tree(y, x, LDS, nu, ancestor_weights, K, num_hp, epochs, batch_size, LR, temper):
    """

    :param y:
    :param x:
    :param LDS:
    :param nu:
    :param ancestor_weights:
    :param K:
    :param HP:
    :param path_LDS:
    :param max_epoch:
    :param batch_size:
    :param LR:
    :param temper:
    :return:
    """
    nT = int(x[:, 0].size)  # Number of trajectories
    rows, cols = x.T.shape
    input_data = torch.from_numpy(x.T).double()
    output_data = torch.from_numpy(y.T).double()
    lds = Variable(torch.from_numpy(LDS), requires_grad=True).double()
    nu = Variable(torch.from_numpy(nu), requires_grad=True).double()
    prev_weights = torch.from_numpy(ancestor_weights).double()

    # Construct optimizer object
    #    optimizer = optim.SGD( [LD, hp], lr = LR,  momentum=0.95, dampening = 0, nesterov = True )
    optimizer = optim.Adam([lds, nu], lr=LR)
    # Perform optimization
    for epoch in tqdm(range(epochs)):
        "Create mini batches"
        batch_idx = create_batches(batch_size, nT)
        for idx in batch_idx:
            optimizer.zero_grad()
            X = input_data[:, idx]
            Y = output_data[:, idx]
            anc_weights = prev_weights[:, idx]
            leaf_weights = torch.zeros(K, idx.size)
            y_leafs = torch.zeros(rows - 1, idx.size, K)
            # if n == num_batches - 1:
            #     X = Variable(input_data[:, n * batch_size:])
            #     Y = Variable(output_data[:, n * batch_size:])
            #     anc_weights = Variable(prev_weights[:, n * batch_size:])
            #     weights_local = Variable(torch.from_numpy(np.zeros((K, len(input_data[0, n * batch_size:])))))
            #     y_local = Variable(torch.from_numpy(np.zeros((rows - 1, len(input_data[0, n * batch_size:]), K))))
            #
            # else:
            #     X = Variable(input_data[:, n * batch_size:(n + 1) * batch_size])
            #     Y = Variable(output_data[:, n * batch_size:(n + 1) * batch_size])
            #     anc_weights = Variable(prev_weights[:, n * batch_size:(n + 1) * batch_size])
            #     weights_local = Variable(torch.from_numpy(np.zeros((K, batch_size))))
            #     y_local = Variable(torch.from_numpy(np.zeros((rows - 1, batch_size, K))))

            # Compute weight of each path
            counter = 0
            for h in range(0, HP):
                leaf_weights[counter, :] = torch.mul(anc_weights[counter, :],
                                                      torch.sigmoid(temper * torch.matmul(X.transpose(0, 1), hp[:, h])))
                leaf_weights[counter + 1, :] = torch.mul(anc_weights[counter + 1, :], torch.sigmoid(
                    -temper * torch.matmul(X.transpose(0, 1), hp[:, h])))
                counter += 2

            # Compute weighted sum of LDS
            for k in range(0, K):
                y_leafs[:, :, k] = torch.mul(leaf_weights[k, :], torch.matmul(lds[:, :, k], X))

            y_pred = torch.sum(y_leafs, 2)

            # Compute difference
            resid = Y - y_pred

            # Compute MSE
            loss = 0.5 * torch.matmul(resid, resid.transpose(0, 1)).trace() / idx.size

            # Perform backprop
            loss.backward()

            # Update parameters
            optimizer.step()

    return lds, nu, resid.detach().numpy(), loss.item()


# In[13]:
def projection(xreal, xinferr):
    Xreals = np.hstack(xreal).T
    Xrot = np.hstack(xinferr).T
    Xrot = np.hstack((Xrot, np.ones((Xrot[:, 0].size, 1))))
    transform = np.linalg.lstsq(Xrot, Xreals)[0].T
    return transform

# In[]:
def generate_trajectory(A, Q, R, starting_pt, depth, leaf_path, K, T, D_in, noise=True, u=None, D_bias=None):
    if u is D_bias is None:
            u = np.ones((1, T))
            D_bias = 1
    x = np.zeros((D_in, T + 1))
    x[:, 0] = starting_pt
    z = np.zeros(T + 1).astype(int)
    for t in range(T):
        log_p = compute_leaf_log_prob(R, x[:, t], K, depth, leaf_path)
        p_unnorm = np.exp(log_p - np.max(log_p))
        p = p_unnorm/np.sum(p_unnorm)
        if noise:  # Stochastically choose the discrete latent and add noise to continuous latent
                choice = npr.multinomial(1, p.ravel(), size=1)
                z[t] = np.where(choice[0, :] == 1)[0][0].astype(int)
                x[:, t + 1] = (A[:, :-D_bias, z[t]] @ x[:, t][:, na] +  \
                              A[:, -D_bias:, z[t]] @ u[:, t][:, na] + \
                              npr.multivariate_normal(np.zeros(D_in), Q[:, :, z[t]])[:, na]).flatten()

        else:  # Use Bayes classifier to choose discrete latent state and add no noise to continuous latent states
            z[t] = np.argmax(choice)
            x[:, t + 1] = (A[:, :-D_bias, z[t]] @ x[:, t][:, na] + \
                           A[:, -D_bias:, z[t]] @ u[:, t][:, na]).flatten()

    log_p = compute_leaf_log_prob(R, x[:, -1], K, depth, leaf_path)
    p_unnorm = np.exp(log_p - np.max(log_p))
    p = p_unnorm / np.sum(p_unnorm)
    choice = npr.multinomial(1, p.ravel(), size=1)
    z[-1] = np.where(choice[0, :] == 1)[0][0]
    return x, z

# In[]:
def MAP_dynamics(x, u, z, Ainit, Qinit, nux, lambdax, Mx, Vx, scale, leaf_nodes, K, depth, no_samples):
    A_est = []
    Q_est = []
    At = copy.deepcopy(Ainit)
    Qt = copy.deepcopy(Qinit)
    for m in tqdm(range(no_samples)):
        At, Qt = sample_leaf_dynamics(x, u, z, At, Qt, nux, 
                                            lambdax, Mx, Vx, scale, leaf_nodes)
        At = sample_internal_dynamics(At, scale, Mx, Vx, depth)
        if m > no_samples/2:
            A_est.append(copy.deepcopy(At))
            Q_est.append(copy.deepcopy(Qt))
    
    #Take average of samples
    Z = len(A_est)
    #Take sample mean as estimate
    for d in range(depth):
        for node in range(2**d):
            At[d][:,:,node] = A_est[0][d][:,:,node]/Z
    Qt = Q_est[0]/Z
    for sample in tqdm(range(1, len(A_est))):
        for k in range(K):
            Qt[:,:,k] += Q_est[sample][:,:,k]/Z
        #Take sample mean as estimate
        for d in range(depth):
            for node in range(2**d):
                At[d][:,:,node] += A_est[sample][d][:,:,node]/Z
    return At, Qt

# In[]:
def gaussian_kernel_smoother(y, sigma, window):
    b = gaussian(window, sigma)
    y_smooth = np.zeros(y.shape)
    neurons = y[:, 0].size
    for neuron in range(neurons):
        y_smooth[neuron, :] = filters.convolve1d(y[neuron, :], b)/b.sum()
    return y_smooth