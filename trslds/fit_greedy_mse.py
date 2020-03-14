import numpy as np
import numpy.random as npr
from . import utils


def top_to_bottom(data, max_depth, max_epoch, batch_size, lr, u=None):
    dx = data[0][:, 0].size  # Dimension of latent space
    hier_LDS = []  # Create list that will store Linear Dynamics per level in the hierarchy
    hier_nu = []
    losses = []

    # In[]
    x_ols = np.hstack([data[idx][:, :-1] for idx in range(len(data))]).T
    y_ols = np.hstack([data[idx][:, 1:] for idx in range(len(data))]).T

    if u is None:
        du = 1
        u_ols = np.ones((1, x_ols[:, 0].size))
    else:
        u_ols = np.hstack(u)
        du = u_ols[:, 0].size
    x_ols = np.hstack((x_ols, u_ols.T))
# In[]:
    "For the root node, we can easily find the LDS that minimizes MSE using the OLS estimator"
    beta = np.linalg.solve((x_ols.T @ x_ols), x_ols.T @ y_ols)
    hier_LDS.append(np.expand_dims(np.array(beta.T), axis=2))
    
    y_ols = y_ols - x_ols @ beta  # Send residual to be fit by trees deeper down the hierarchy
    losses.append(np.trace(y_ols.T @ y_ols) / y_ols[:, 0].size)
    del beta

    # In[]
    for level in range(1, max_depth):
        print(level)
        K = int(2 ** level)  # number of nodes at current level

        "Initialize parameters of dynamics at current level"
        lds = 1e-5 * npr.rand(dx, dx + du, K) - 1e-5 / 2

        num_hp = int(2 ** (level - 1))  # number of hyperplanes that need to be learned

        "Initialize hyperplanes"
        nu = np.zeros((dx + 1, num_hp))
        nu[:-1, :] = npr.randn(dx, num_hp)
        
# In[4]:
        "Compute weights of ancestral path"
        if level == 1:
            ancestor_weights = np.ones((K, x_ols[:, 0].size))
        else:
            ancestor_weights = np.ones((K, x_ols[:, 0].size))
            for j in range(level - 1):
                hyper_planes = np.array(hier_nu[j])
                counter = 0
                temp_array = np.zeros((2 * hyper_planes[0, :].size, x_ols[:, 0].size))
                for k in range(hyper_planes[0, :].size):
                    temp_array[counter, :] = 1/(1+np.exp(-np.matrix(hyper_planes[:, k]) * x_ols.T))
                    temp_array[counter + 1, :] = 1 - temp_array[counter, :]
                    counter += 2
                ancestor_weights = np.multiply(ancestor_weights, np.repeat(temp_array, int(K/temp_array[:, 0].size),
                                                                           axis=0))

        # Optimize
        lds, nu, y_ols, loss = utils.optimize_tree(y_ols, x_ols, lds, nu, ancestor_weights, K, num_hp, max_epoch,
                                       batch_size, lr, 1)
        losses = losses + loss
        hier_LDS.append(np.array(lds.data.numpy()))
        if level != 0:
            hier_nu.append(np.array(nu.data.numpy()))
    
    return hier_LDS, hier_nu, losses
       

# In[]:
def initialize_discrete(X, R, depth, K, leaf_path, random=False):
    """
    xs=continuous latent states
    R,r= hpyer planes
    depth=depth of tree
    K = number of leaf nodes
    leaf_paths = paths associated with each leaf
    """
    
    Z = []
    Path = []
    for idx in range(len(X)):
        z = np.zeros(X[idx][0, :].size)
        path = np.zeros((depth, X[idx][0, :].size))
        for t in range(X[idx][0, :].size):
            log_prob = utils.compute_leaf_log_prob(R, X[idx][:, t], K, depth, leaf_path)
            p = np.exp(log_prob - np.max(log_prob))
            p = p/np.max(p)

            if random:  # If true then sample from pmf defined by p
                choice = npr.multinomial(1, p, size=1)

                path[:, t] = leaf_path[:, np.where(choice[0, :] == 1)[0][0]].ravel()
                z[t] = np.where(choice[0, :] == 1)[0][0]
            else:  # Initialize with bayes classifier
                choice = np.argmax(p)
                path[:, t] = leaf_path[:, choice].ravel()
                z[t] = choice

        Z.append(z)
        Path.append(path)
    return Z, Path

