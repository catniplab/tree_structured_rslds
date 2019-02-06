import numpy as np
import numpy.random as npr
import utils
def initialize_dynamics( data, max_depth, max_epoch, batch_size, lr ):
    D_in = data[0][:, 0].size  # Dimension of latent space

    Hier_LDS=[] #Create list that will store Linear Dynamics per level in the hierarchy
    Hier_nu=[]
    
# In[1]:
    "For the root node, we can easily find the LDS that minimizes MSE using the  OLS estimator"
    x_ols = np.hstack([data[idx][:, :-1] for idx in range(len(data))]).T
    y_ols = np.hstack([data[idx][:, 1:] for idx in range(len(data))]).T - x_ols
    x_ols = np.hstack((x_ols, np.ones((x_ols[:, 0].size, 1))))

    # Get the OLS estimate
    beta = np.linalg.solve((x_ols.T @ x_ols), x_ols.T @ y_ols).T
    Hier_LDS.append(np.expand_dims(np.array(beta), axis=2))
    
    del beta

# In[2]:
    "Start of Optimization"
    for level in range(1, max_depth):
        print(level)
        
# In[3]:
        "Initialize parameters"
        K = int(2**level)  # Number of nodes at current level
        LDS = np.zeros((D_in, D_in + 1, K))  # Used to hold dynamics at each node
        
        HP = int(2**(level - 1))  # Number of hyperplanes that need to be learned
        nu = np.zeros((D_in + 1, HP))
                
        #Initalize LDS
        for j in range(K):
            LDS[:, :, j] = 1e-5*np.random.rand(D_in, D_in + 1) - 5e-6
        
        #Initalize Hyperplanes
        for j in range(HP):
            #Draw hyper-planes from very broad prior
            temp = np.matrix(npr.multivariate_normal(np.zeros(D_in), 20*np.eye(D_in)))
            #Normalize to obtain only direction
            nu[:D_in, j] = 1e-3*temp / np.sqrt(temp*temp.T)


        # Compute the sum of the LDS's that have occurred before the current depth in the tree
        LDS_path = np.zeros((D_in, D_in + 1, K))
        
        for slice in range(level):
            temp_LDS = np.zeros((D_in, D_in + 1, int(2**slice)))
            for j in range(int(2**slice)):
                temp_LDS[:, :, j] = Hier_LDS[slice][:, :, j]
            
            temp_LDS = np.repeat(temp_LDS, int(K/2**slice), axis=2)
            
            for j in range(K):
                LDS_path[:, :, j] += temp_LDS[:, :, j]
        
# In[4]:
        "Compute weights of ancestral path"
        if level == 1:
            ancestor_weights = np.ones((K, x_ols[:, 0].size))
        else:
            ancestor_weights = np.ones((K, x_ols[:, 0].size))
            for j in range(level - 1):
                hyper_planes = np.array(Hier_nu[j])
                counter = 0
                temp_array = np.zeros((2 * hyper_planes[0, :].size, x_ols[:, 0].size))
                for k in range(hyper_planes[0, :].size):
                    temp_array[counter, :] = 1/(1+np.exp(-np.matrix(hyper_planes[:, k]) * x_ols.T))
                    temp_array[counter + 1, :] = 1 - temp_array[counter, :]
                    counter += 2
                ancestor_weights = np.multiply(ancestor_weights, np.repeat(temp_array, int(K/temp_array[:, 0].size), axis =0))

        #Optimzie
        LDS, nu = utils.optimize_tree(y_ols, x_ols, LDS, nu, ancestor_weights, K, HP, LDS_path, max_epoch,
                                       batch_size, lr, 1)

        Hier_LDS.append(np.array(LDS.data.numpy()))
        if level != 0:
            Hier_nu.append(np.array(nu.data.numpy()))
    
    return Hier_LDS, Hier_nu
       

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

