import numpy as np
import numpy.random as npr
from tqdm import tqdm
from trslds.models import TroSLDS
from numpy import newaxis as na
from trslds import utils
import matplotlib.pyplot as plt
from trslds import initialize as init
from trslds import plotting
import seaborn as sns
color_names = ["windows blue", "leaf green","red", "orange"]
colors_leaf = sns.xkcd_palette(color_names)
npr.seed(0)



def resample(no_samples, trslds):
    trslds._initialize_polya_gamma()  # Initialize polya-gamma rvs
    for m in tqdm(range(no_samples)):
        trslds._sample_emission()  # sample emission parameters
        trslds._sample_hyperplanes()  # sample hyperplanes
        trslds._sample_dynamics()  # Sample dynamics of tree
        trslds._sample_discrete_latent()  # Sample discrete latent states
        trslds._sample_continuous_latent()  # Sample continuous latent state



# In[]:
# Load in dataset
iclr_lorenz = np.load('iclr_lorenz.npy', allow_pickle=True)[()]
#Extract out the observations and latent states
Xtrue = iclr_lorenz['X']
Y = iclr_lorenz['Y']

D_out = Y[0][:, 0].size #Obtain dimension of observation space
D_in = 3  #Dimension of latent space
K = 4 #Number of discrete latent states

# In[]:
# Initialize the model
max_epochs = 200
batch_size = 128
lr = 0.0001
A, C, R, X, Z, Path, possible_paths, leaf_path, leaf_nodes = init.initialize(Y, D_in, K, max_epochs, batch_size,
                                                                             lr)
Qstart = np.repeat(np.eye(D_in)[:, :, na], K, axis=2)
Sstart = np.eye(D_out)

kwargs = {'D_in': D_in, 'D_out': D_out, 'K': K, 'dynamics': A, 'dynamics_noise': Qstart, 'emission': C,
          'emission_noise': Sstart,
          'hyper_planes': R, 'possible_paths': possible_paths, 'leaf_path': leaf_path, 'leaf_nodes': leaf_nodes,
          'scale': 0.8}
trslds = TroSLDS(**kwargs) #Instantiiate the model


#Add data to model
for idx in range(len(Y)):
    trslds._add_data(X[idx], Y[idx], Z[idx], Path[idx])


#Perform Gibbs to train the model
no_samples = 200
trslds = resample(no_samples, trslds)

