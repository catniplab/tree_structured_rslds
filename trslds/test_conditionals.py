import numpy as np
import copy
import numpy.random as npr
import matplotlib.pyplot as plt
from numpy import newaxis as na
import utils
import conditionals as samp
from tqdm import tqdm
npr.seed(0)
from scipy.stats import mode
# In[0]:
dim = 2
dim_y = 2
nu = dim + 1
K = 4
depth = 3
"""
Parameters of generative model
"""
depth=3
scale=0.6
dim=2
Q=.001*np.eye(dim)

"""
Hyper-planes
"""
R_par=np.zeros((dim + 1,1))
R_par[0,0]=100
R_par[1,0]=100
r_par=np.array([0.0])

R=[]
R.append(copy.deepcopy(R_par))
R_temp=np.zeros((dim + 1,2))
R_temp[:-1,0]=np.array([-100, 100]) #Left hyperplane
R_temp[:-1,1]=np.array([-100, 100]) #Right hyperplane
R.append(copy.deepcopy(R_temp))


"""
Emission parameters
"""
C=np.eye(dim)
C[0, 1] = 2
S=.001*np.eye(dim)

"""
Dynamic Parameters
"""
A = np.zeros((dim, dim + 1, K))

#layer 2
theta_1 = -.15*np.pi/2
theta_2 = -.05*np.pi/2

A[:, :-1, 0] = np.eye(dim)
A[:, -1, 0] = np.array( [.25, 0] )

A[:, :-1, 1] = np.array( [ [ np.cos( theta_1 ), -np.sin( theta_1 ) ], [ np.sin( theta_1 ), np.cos( theta_1 ) ] ] )
A[:, -1, 1] = ((-A[:, :-1, 1] + np.eye(dim) ) @ np.array ( [4, 0] )[:, na]).flatten()

A[:, :-1, 2] = np.array( [ [ np.cos( theta_2 ), -np.sin( theta_2 ) ], [ np.sin( theta_2 ), np.cos( theta_2 ) ] ] )
A[:, -1, 2] = ((-A[:, :-1, 2] + np.eye(dim) ) @ np.array ( [-4, 0] )[:, na]).flatten()


A[:, :-1, 3] = np.eye( dim )
A[:, -1, 3] = np.array( [-.05, 0] )



possible_paths=np.ones((depth,K))
for d in range(1,depth):
    temp=np.arange(0,2**int(d))+1
    possible_paths[d,:]=np.repeat(temp,int(K/temp.size))
    
# In[1]:
#Generate data
no_realizations = 50
Tmax = 800
Tmin = 400

X = []
Y = []
paths = []
Z = []

starting_pts = np.random.uniform(-10, 10, (dim, no_realizations) ) 
for n in tqdm(range(no_realizations)):
    T = npr.randint(Tmin, Tmax + 1)
    #Will be used to keep track of the path at every time instant
    path=np.ones( (depth, T+1) )
    #Generate data
    x = np.zeros((dim,T+1))
    x[:,0] = starting_pts[:, n]
    y = np.zeros((dim,T))
    z = K * np.ones(T + 1)
    for t in range(T):
        #First decide what path you will take
        log_prior_prob=np.zeros((K,1))
        for d in range(0,depth-1):
            temp=np.zeros((int(2**(d+1)),1)) #Make an array whose length is equal to the number hyperplanes at that level of the tree
            counter=0
            for j in range(0,int(2**d)):
                v = np.matmul(R[d][:-1, j], x[:, t]) + R[d][-1, j]
                temp[counter]=np.log(utils.sigmoid(v))
                temp[counter+1]=np.log(utils.sigmoid(-v))
                counter+=2
            log_prior_prob=log_prior_prob+np.repeat(temp,int(K/temp.size),axis=0)
        p_norm=np.exp(log_prior_prob)
        
        #Sample which path we will take
        choice = np.random.multinomial(1,p_norm.ravel(),size=1)
        path[:, t] =  possible_paths[:, np.where(choice==1)[1][0]].ravel()
        z[t] = np.where(choice[0, :] == 1)[0][0]
        
        #Simulate forward
        x[:, t + 1] = (A[:, :-1, int(z[t])] @ x[:, t][:, na] + A[:, -1, int(z[t])][:, na] 
                    + np.random.multivariate_normal( [0,0], Q )[:, na]).flatten()
        y[:, t] = (C @ x[:, t + 1][:, na] + np.random.multivariate_normal( [0,0], S )[:, na]).flatten()
    
    log_prior_prob=np.zeros((K,1))
    for d in range(0,depth-1):
        temp=np.zeros((int(2**(d+1)),1)) #Make an array whose length is equal to the number hyperplanes at that level of the tree
        counter=0
        for j in range(0,int(2**d)):
            v = np.matmul(R[d][:-1, j], x[:, T]) + R[d][-1, j]
            temp[counter]=np.log(utils.sigmoid(v))
            temp[counter+1]=np.log(utils.sigmoid(-v))
            counter+=2
        log_prior_prob=log_prior_prob+np.repeat(temp,int(K/temp.size),axis=0)
    p_norm=np.exp(log_prior_prob)
    
    #Sample which path we will take
    choice = np.random.multinomial(1,p_norm.ravel(),size=1)
    path[:, T] =  possible_paths[:, np.where(choice==1)[1][0]].ravel()
    z[T] = np.where(choice[0, :] == 1)[0][0]

    X.append(x)
    Y.append(y)
    paths.append(path)
    Z.append(z)
#     X.append(copy.copy(x))
#     Y.append(copy.copy(y))
#     paths.append(copy.copy(path))
#     Z.append(copy.copy(z))
    
# In[2]:
"Plot data"
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
for idx in tqdm(range(no_realizations)):
    ax.scatter(X[idx][0, np.where(Z[idx]==0)], X[idx][1, np.where(Z[idx]==0)], color='green')
    ax.scatter(X[idx][0, np.where(Z[idx]==1)], X[idx][1, np.where(Z[idx]==1)], color='red')
    ax.scatter(X[idx][0, np.where(Z[idx]==2)], X[idx][1, np.where(Z[idx]==2)], color='blue')
    ax.scatter(X[idx][0, np.where(Z[idx]==3)], X[idx][1, np.where(Z[idx]==3)], color='purple')
#    ax.scatter(X[idx][0, 0], X[idx][1, 0], marker='x', s=400)

# In[3]:
"Define hyperparameters used in sampling"
#Emission
My = np.zeros((dim_y, dim + 1))
My[:, :-1] = np.eye(dim_y)
nuy = dim_y + 1
lambday = 1e-7 * np.eye(dim_y)
Vy = 1000 * np.eye(dim + 1)

#Dynamics
Mx = np.zeros((dim, dim + 1))
Mx[:, :-1] = np.eye(dim)
nux = dim + 1
lambdax =1e-100000 * np.eye(dim)
Vx = 1000 * np.eye(dim + 1)

#Hyperplanes
mu = np.zeros(dim + 1)
Sigma = 10000 * np.eye(dim + 1)

#boolean mask
mask = [np.ones(X[idx][0, 1:].size).astype(bool) for idx in range(no_realizations)]

leaf_nodes = [] #A list of tuples (d,n,k) where d and n are the depth and node in the tree respectively
leaf_path = possible_paths
for d in range( depth-2, depth ): #Check the last two levels
    for k in range(K):
        if d == depth -1: #bottom level of tree
            if not np.isnan(leaf_path[d, k]):
                leaf_nodes.append( (int(d), int(leaf_path[d,k]-1), int(k) ) )
        else: #level before bottom level
            if np.isnan(leaf_path[d+1,k]):
                leaf_nodes.append( ( int(d), int(leaf_path[d, k]-1), int(k) ) )

# In[4]:
"Emission parameters"
#no_samples = 1000
#C_samples = np.ones((dim_y, dim + 1, no_samples))
#S_samples = np.ones((dim_y, dim_y, no_samples))
#for m in tqdm(range(no_samples)):
#    C_samples[:, :, m], S_samples[:, :, m] = samp.emission_parameters(Y, X, mask, nuy, lambday, My, Vy, normalize=False)
#
#Cest = np.mean(C_samples, axis=2)
#Sest = np.mean(S_samples, axis=2)

# In[5]:
"Leaf dynamics"
Aplace = []
for level in range(depth):
    Aplace.append(np.zeros((dim, dim + 1, 2**level)))
Qplace = np.zeros((dim, dim, K))
no_samples = 1000
Asamples = np.zeros((dim , dim + 1, K, no_samples))
Qsamples = np.zeros((dim, dim, K, no_samples))
scale = 0.9
U = [np.ones((1, X[idx][0, :].size)) for idx in range(len(X))]

leaf_nodes = [] #A list of tuples (d,n,k) where d and n are the depth and node in the tree respectively
leaf_path = possible_paths
for d in range( depth-2, depth ): #Check the last two levels
    for k in range(K):
        if d == depth -1: #bottom level of tree
            if not np.isnan(leaf_path[d, k]):
                leaf_nodes.append( (int(d), int(leaf_path[d,k]-1), int(k) ) )
        else: #level before bottom level
            if np.isnan(leaf_path[d+1,k]):
                leaf_nodes.append( ( int(d), int(leaf_path[d, k]-1), int(k) ) )

for m in tqdm(range(no_samples)):
    Aplace, Qsamples[:, :, :, m] = utils.sample_leaf_dynamics(X, U, Z, Aplace, 
                    Qplace, nux, lambdax, Mx, Vx, scale, leaf_nodes)
    Aplace = utils.sample_internal_dynamics(Aplace, scale, Mx, Vx, depth)
    Asamples[:, :, :, m] = Aplace[-1]
Aest = np.mean(Asamples, axis=3)
Qest = np.mean(Qsamples, axis=3)
print('\n')
[print(Qest[:, :, k]) for k in range(K)]

for k in range(K):
    print(Aest[:, :, k])
    print(A[:, :, k])

# In[6]:
"Test hyperplanes"
#no_samples = 5000
#Rroot = np.zeros((dim + 1, 1, no_samples))
#Rsecond = np.zeros((dim + 1, 2, no_samples))
#
#R_place = []
#for level in range(depth - 1):
#    R_place.append(2*npr.rand(dim + 1, 2**level) - 1)
##R_place = copy.deepcopy(R)
#omega = [ np.zeros((depth - 1, X[idx][0, :].size)) for idx in range(len(X)) ]
#for m in tqdm(range(no_samples)):
#    omega = samp.pg_tree_posterior(X, omega, R_place, paths, depth)
#    R_place = utils.sample_hyperplanes(X, omega, paths, depth, mu, Sigma, 
#                                       possible_paths, R_place)
#    Rroot[:, :, m] = R_place[0]
#    Rsecond[:, :, m] = R_place[1]
#R1est = np.mean(Rroot[:, :, int(no_samples/2):], axis=2)
#R2est = np.mean(Rsecond[:, :, int(no_samples/2):], axis=2)

# In[7]:
"Test discrete latent states"
no_samples = 100
Z_samples = []
Ztemp = copy.deepcopy(Z)
path_temp = copy.deepcopy(paths)
U = [np.ones((1, X[idx][0, :].size)) for idx in range(len(X))]

for m in tqdm(range(no_samples)):
    Ztemp, path_temp = samp.discrete_latent_recurrent_only(Ztemp, path_temp, leaf_path, 
                                           K, X, U, A, np.repeat(Q[:, :, na], K, axis=2),
                                           R, depth, 1)
    Z_samples.append(Ztemp)


# In[8]:
zest = []
for idx in tqdm(range(no_realizations)):
    ztemp = np.zeros((no_samples, Z[idx].size))
    for m in range(no_samples):
        ztemp[m, :] = Z_samples[m][idx]
    zest.append(mode(ztemp, axis = 0)[0])

correct = 0
total = 0
for idx in range(no_realizations):
    correct += np.sum(zest[idx] == Z[idx])
    total += zest[idx].size

print(correct/total)

# In[8]:
"Test continuous latent states"
no_samples = 10
X_samples = []
Ct = np.hstack((C, np.zeros((2, 1))))
Xtemp = copy.deepcopy(X)
U = [np.ones((1, X[idx][0, :].size)) for idx in range(len(X))]
max_len = max([Z[idx].size for idx in range(no_realizations)])
P = np.repeat(20*np.eye(dim)[:, :, na], max_len, axis=2)
alphas = np.zeros((dim, max_len))
omega = [ np.zeros((depth - 1, X[idx][0, :].size)) for idx in range(len(X)) ]
Lambdas = np.repeat(20*np.eye(dim)[:, :, na], max_len, axis=2)
for m in tqdm(range(no_samples)):
    omega = samp.pg_tree_posterior(Xtemp, omega, R, paths, depth)
    Xtemp = samp.pg_kalman(dim, 1, Xtemp, U, P, A, np.repeat(Q[:, :, na], K, axis=2), 
                           Ct, S, Y, paths, Z, omega, alphas, Lambdas, R, depth)
    X_samples.append(Xtemp)
# In[9]:
Xest = []
for idx in tqdm(range(no_realizations)):
    xtemp = np.zeros((dim, X[idx][0, :].size, no_samples))
    for m in range(no_samples):
        xtemp[:, :, m] = X_samples[m][idx]
    Xest.append(np.mean(xtemp, axis=2))

fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

for idx in range(no_realizations):
    #Dimension 1
    ax1.plot(np.arange(X[idx][0, :].size), X[idx][0, :])
    ax1.plot(np.arange(X[idx][0, :].size), Xest[idx][0, :], linestyle='--')
    
    #Dimension 2
    ax2.plot(np.arange(X[idx][0, :].size), X[idx][1, :])
    ax2.plot(np.arange(X[idx][0, :].size), Xest[idx][1, :], linestyle='--')

# In[9]:
"Create spike trains for testing conditionals"
npr.seed(10)
neurons = 20
Cspike = npr.rand(neurons, dim + 1) - 0.1
spikes = []
V =[]
Prob_of_spike = []
for idx in tqdm(range(no_realizations)):
    V.append(Cspike @ np.vstack((X[idx][:, 1:], np.ones((1, X[idx][0, 1:].size)))))
    Prob_of_spike.append(1 / (1 + np.exp(-V[idx])))
    spikes.append(npr.binomial(1, Prob_of_spike[idx]))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.imshow(spikes[0], cmap = 'binary')
#ax.set_aspect('auto')
plt.tight_layout()
#
## In[10]:
#"Test to see if spike emission conditionals are working"
#no_samples = 100
#C_samples = np.zeros((neurons, dim + 1, no_samples))
#Ctemp = np.zeros(Cspike.shape)
#omegay = [np.zeros(spikes[idx].shape) for idx in range(no_realizations)]
#mask = [np.ones(X[idx][0, 1:].size).astype(bool) for idx in range(no_realizations)]
#omegay = samp.pg_spike_train(X, Cspike, omegay, neurons)
#for m in tqdm(range(no_samples)):
#    Ctemp = samp.emission_parameters_spike_train(spikes, X, omegay, mask, 
#                                                 mu, Sigma, normalize=False)
#    C_samples[:, :, m] = Ctemp
#    omegay = samp.pg_spike_train(X, Ctemp, omegay, neurons)
#    
#
## In[11]:
#Cest = np.mean(C_samples[:, :, int(no_samples/2):], axis=2)
#est_prob = []
#for idx in tqdm(range(no_realizations)):
#    V = Cest @ np.vstack((X[idx][:, 1:], np.ones((1, X[idx][0, 1:].size))))
#    est_prob.append(1 / (1 + np.exp(-V)))
#
#fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(111)
#for neuron in range(neurons):
#    T = spikes[0][0, :].size
#    ax.plot(np.arange(T), Prob_of_spike[0][neuron, :])
#    ax.plot(np.arange(T), est_prob[0][neuron, :], linestyle='--')

# In[8]:
"Test spike continuous latent states"
no_samples = 30
X_samples = []
Xtemp = copy.deepcopy(X)
U = [np.ones((1, X[idx][0, :].size)) for idx in range(len(X))]
max_len = max([Z[idx].size for idx in range(no_realizations)])
P = np.repeat(20*np.eye(dim)[:, :, na], max_len, axis=2)
alphas = np.zeros((dim, max_len))
omega = [ np.zeros((depth - 1, X[idx][0, :].size)) for idx in range(len(X)) ]
omegay = [ np.zeros(Y[idx].shape) for idx in range(len(X)) ]
Lambdas = np.repeat(20*np.eye(dim)[:, :, na], max_len, axis=2)
for m in tqdm(range(no_samples)):
    omegay = samp.pg_spike_train(X, Cspike, omegay, neurons)
    omega = samp.pg_tree_posterior(Xtemp, omega, R, paths, depth)
    Xtemp = samp.pg_kalman_spike(dim, 1, Xtemp, U, P, A, np.repeat(Q[:, :, na], K, axis=2), 
                           Cspike, spikes, paths, Z, omega, omegay, alphas, Lambdas, R, depth)
    X_samples.append(Xtemp)
# In[9]:
Xest = []
for idx in tqdm(range(no_realizations)):
    xtemp = np.zeros((dim, X[idx][0, :].size, no_samples))
    for m in range(no_samples):
        xtemp[:, :, m] = X_samples[m][idx]
    Xest.append(np.mean(xtemp, axis=2))

fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

for idx in range(no_realizations):
    #Dimension 1
    ax1.plot(np.arange(X[idx][0, :].size), X[idx][0, :])
    ax1.plot(np.arange(X[idx][0, :].size), Xest[idx][0, :], linestyle='--')
    
    #Dimension 2
    ax2.plot(np.arange(X[idx][0, :].size), X[idx][1, :])
    ax2.plot(np.arange(X[idx][0, :].size), Xest[idx][1, :], linestyle='--')