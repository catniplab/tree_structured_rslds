import numpy as np
import numpy.random as npr
import trslds.conditionals as samp
from tqdm import tqdm
# In[1]:
dim = 2
d_bias = 1
Amean = np.zeros((dim, dim + d_bias))
Amean[:, :-d_bias] = 0.99 * np.eye(dim)
no_child = 100000
scale = 0.5
V = 40 * np.eye(dim + d_bias)
# In[2]:
npr.seed(0)
Areals = np.zeros((dim, dim + d_bias, no_child))
for m in tqdm(range(no_child)):
    Areals[:, :, m] = npr.multivariate_normal(Amean.flatten(order='F'),
                                              np.kron(scale*V, np.eye(dim))).reshape((dim, dim + d_bias), order='F')
Achild = np.sum(Areals, axis=2)

# In[3]:
no_samples = 1000
A_samples = np.zeros((dim, dim + d_bias, no_samples))
for m in tqdm(range(no_samples)):
    A_samples[:, :, m] = samp._internal_dynamics(np.zeros((dim, dim + d_bias)), 
             V, Achild, scale*V, no_child)

Aest = np.mean(A_samples, axis=2)