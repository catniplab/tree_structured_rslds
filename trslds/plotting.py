import numpy as np
import numpy.random as npr
from . import utils
from numpy import newaxis as na
from matplotlib.colors import LinearSegmentedColormap


# In[1]:
def contour_plt(R, xmin, xmax, ymin, ymax, delta, depth, leaf_path, K):
    x, y = np.arange(xmin,xmax, delta), np.arange(ymin,ymax, delta)
    X,Y = np.meshgrid(x,y)
    points = np.zeros( (X[:,0].size, X[0,:].size, 2) )
    points[:,:,0] = X
    points[:,:, 1] = Y
    colors = np.zeros( ( X[:,0].size, X[0,:].size, K   ) )
    
    for rows in range( X[:,0].size):
        for cols in range( X[0,:].size):
            pts = points[rows,cols,:] 
            #Compute log of probabilities
            log_prior_prob = utils.compute_leaf_log_prob(R, pts, K, depth, leaf_path)
            p_unnorm = np.exp( log_prior_prob - np.max(log_prior_prob) )
            p_norm = p_unnorm / np.sum(p_unnorm)
            colors[rows,cols,:] = np.array(p_norm).flatten()
    return x, y, colors


# In[2]:
def rot_contour_plt(R, xmin, xmax, ymin, ymax, delta, depth, leaf_path, K, transform):
    x, y = np.arange(xmin, xmax, delta), np.arange(ymin, ymax, delta)
    X, Y = np.meshgrid(x, y)
    points = np.zeros((X[:, 0].size, X[0, :].size, 2))
    points[:, :, 0] = X
    points[:, :, 1] = Y
    colors = np.zeros((X[:, 0].size, X[0, :].size, K))

    for rows in range(X[:, 0].size):
        for cols in range(X[0, :].size):
            og_pt = points[rows, cols, :]

            #Transform from real space to inferred space
            pts = np.linalg.solve(transform[:,:-1],og_pt-transform[:,-1])
            #Compute log of probabilities
            log_prior_prob = utils.compute_leaf_log_prob(R, pts, K, depth, leaf_path)
            p_unnorm = np.exp(log_prior_prob - np.max(log_prior_prob))
            p_norm = p_unnorm / np.sum(p_unnorm)
            colors[rows, cols, :] = np.array(p_norm).flatten()
    return x, y, colors


# In[3]:
def vector_field(Aleaf, R, xmin, xmax, ymin, ymax, delta, depth, leaf_path, K):
    x, y = np.arange(xmin,xmax, delta), np.arange(ymin,ymax, delta)
    X,Y = np.meshgrid(x,y)
    points = np.zeros( (X[:,0].size, X[0,:].size, 2) )
    points[:,:,0] = X
    points[:,:, 1] = Y
    arrows = np.zeros( points.shape )

    for rows in range( X[:,0].size):
        for cols in range( X[0,:].size):
            pts = points[rows, cols, :].T

            #Compute log of probabilities
            log_prior_prob = utils.compute_leaf_log_prob(R, pts, K, depth, leaf_path)
            p_unnorm = np.exp( log_prior_prob - np.max(log_prior_prob) )
            p_norm = p_unnorm / np.sum(p_unnorm)

            arrow = 0
            for k in range(K):
                arrow += p_norm[k] *(Aleaf[:, :-1, k] @ pts[:, na] + Aleaf[:, -1, k][:, na]).ravel()

            arrows[rows, cols, :] = arrow-np.array(pts).flatten()

    return x, y, arrows

# In[4]:
def rot_vector_field(Aleaf,R, xmin, xmax, ymin, ymax, delta, depth, leaf_path, K, transform):
    x, y = np.arange(xmin, xmax, delta), np.arange(ymin, ymax, delta)
    X, Y = np.meshgrid(x, y)
    points = np.zeros((X[:, 0].size, X[0, :].size, 2))
    points[:, :, 0] = X
    points[:, :, 1] = Y
    arrows = np.zeros(points.shape)

    for rows in range(X[:, 0].size):
        for cols in range(X[0, :].size):
            og_pt = points[rows, cols, :]

            #Project model from real space to inferred space
            pts = np.linalg.solve(transform[:, :-1], og_pt - transform[:, -1])
            #Compute log of probabilities
            log_prior_prob = utils.compute_leaf_log_prob(R, pts, K, depth, leaf_path)
            p_unnorm = np.exp(log_prior_prob - np.max(log_prior_prob))
            p_norm = p_unnorm / np.sum(p_unnorm)

            arrow = 0
            for k in range(K):
                arrow += (p_norm[k] * (Aleaf[:, :-1, k] @ pts[:, na] + Aleaf[:, -1, k][:, na])).ravel()

            #Project onto real space
            og_arrow = transform[:, :-1]@arrow + transform[:, -1]

            arrows[rows, cols, :] = np.array(og_arrow - og_pt).flatten()
    return x, y, arrows

# In[14]:
def gradient_cmap(gcolors, nsteps=256, bounds=None):
    """
    Make a colormap that interpolates between a set of colors
    """
    ncolors = len(gcolors)
    if bounds is None:
        bounds = np.linspace(0, 1, ncolors)

    reds = []
    greens = []
    blues = []
    alphas = []
    for b, c in zip(bounds, gcolors):
        reds.append((b, c[0], c[0]))
        greens.append((b, c[1], c[1]))
        blues.append((b, c[2], c[2]))
        alphas.append((b, c[3], c[3]) if len(c) == 4 else (b, 1., 1.))

    cdict = {'red': tuple(reds),
             'green': tuple(greens),
             'blue': tuple(blues),
             'alpha': tuple(alphas)}

    cmap = LinearSegmentedColormap('grad_colormap', cdict, nsteps)
    return cmap