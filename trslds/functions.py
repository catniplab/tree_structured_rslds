#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 02:07:59 2018

@author: user
"""

import torch
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from tqdm import tqdm


"""
Generate trajectory from FitzHugh-Nagumo with noisy input current
"""
def FitzHugh(dt,T,start):
#    dt=0.1
    v=np.zeros(T)
    w=np.zeros(T)
    v[0]=start[0]
    w[0]=start[1]
    
    for t in range(0,T-1):
        I=np.random.normal(0.7,0.2,1)*np.sqrt(dt/0.1)
        dv=(v[t]-np.power(v[t],3)/3-w[t]+I)*dt
        dw=0.08*dt*(v[t]+0.7-0.8*w[t])
        v[t+1]=v[t]+dv
        w[t+1]=w[t]+dw
    
    states=np.zeros((2,T))
    states[0,:]=v
    states[1,:]=w
    return states

"""
Generate trajectory from FitzHugh-Nagumo with noisy input current
"""
def FitzHugh_no_noise(dt,T,start):
#    dt=0.1
    v=np.zeros(T + 1)
    w=np.zeros(T + 1)
    v[0]=start[0]
    w[0]=start[1]
    
    for t in range(0,T):
        I = 0.7
        dv=(v[t]-np.power(v[t],3)/3-w[t]+I)*dt
        dw=0.08*dt*(v[t]+0.7-0.8*w[t])
        v[t+1]=v[t]+dv
        w[t+1]=w[t]+dw
    
    states=np.zeros((2,T + 1))
    states[0,:]=v
    states[1,:]=w
    return states



"""
Caculate gradients for  root node
"""
def calc_root_gradient( y, x, LDS,  max_epoch ):
    #y = output data
    #x = input data
    #LDS = dynamics of root node
    
    T = x[:, 0].size
    
    #Define variables
    X = Variable( torch.from_numpy( x.T ).float( ) )
    Y = Variable( torch.from_numpy( y.T ).float( ) )
    LD = Variable( torch.from_numpy( LDS ).float( ), requires_grad = True )
    
    #Create optimizer object
    optimizer = optim.SGD( [LD], lr = 0.001,  momentum=0.95 )
    
    #Perform optimization
    for epoch in range( max_epoch ):
        print( epoch )
        optimizer.zero_grad()
        
        #Make Prediciton of Y values
        y_pred = torch.matmul(  LD[:,:,0], X )
        
        #Compute difference
        z = Y - y_pred
        
        #Compute MSE
        loss = torch.matmul( z.transpose( 0, 1 ), z ).trace()/T
        
        #Perform backprop
        loss.backward()
        
        #Update parameters
        optimizer.step()
        
    return LD

"""
Caculate gradients for  other nodes in the tree
"""
def calc_HP_gradient( dim, y, x, LDS, nu, ancestor_weights, K, HP, path_LDS, max_epoch, batch_size, LR, T, temper ):
    #dim=dimension of latent space
    #y = output data
    #x = input data
    #LDS= linear dynamics of nodes in current depth of the tree
    #nu= hyperplanes
    #ancestor_weights= prpobability of previous paths
    #K=number of nodes at current depth of the tree
    #HP= number of hypeplanes
    #path_LDS= weighted sum of previous LDS
    #max_epoch = maximum number of epochs 
    #batch = size of batch
    # LR = learning rate
    #temper = parameter in sigmoid function used to make decision boundaries sharper
    
    nT = int( x[:, 0].size ) #Number of trajectories
    
    N = int( np.ceil( nT/ batch_size) )
    batch_size = int( batch_size)
    rows, cols = x.T.shape
    
    input_data = torch.from_numpy( x.T ).double() 
    output_data =  torch.from_numpy( y.T ).double()
    LD = Variable( torch.from_numpy( LDS ), requires_grad = True ).double()
    hp = Variable( torch.from_numpy( nu ), requires_grad = True ).double()
    prev_weights =  torch.from_numpy( ancestor_weights ).double()
    p_LDS = Variable(torch.from_numpy(path_LDS) ).double()

    #Construct optimizer object
#    optimizer = optim.SGD( [LD, hp], lr = LR,  momentum=0.95, dampening = 0, nesterov = True )
    optimizer = optim.Adam([LD, hp], lr = LR)
    #Perform optimization
    for epoch in range( max_epoch ):
        for n in range(N) :
            optimizer.zero_grad()
            
            if n == N-1:
                X = Variable( input_data[:, n*batch_size:] )
                Y = Variable( output_data[:, n*batch_size:] )
                anc_weights = Variable( prev_weights[:,n*batch_size: ] )
                weights_local = Variable( torch.from_numpy( np.zeros( ( K, len( input_data[0,n*batch_size:] ) ) ) ) )
                y_local = Variable( torch.from_numpy(np.zeros( ( rows-1, len( input_data[0,n*batch_size:] ), K ) ) ) )
            
            else:
                X = Variable( input_data[:, n*batch_size:(n+1)*batch_size] )
                Y = Variable( output_data[:, n*batch_size:(n+1)*batch_size] )
                anc_weights = Variable( prev_weights[:,n*batch_size:(n+1)*batch_size ] )
                weights_local = Variable( torch.from_numpy( np.zeros( ( K, batch_size ) ) ) )
                y_local = Variable( torch.from_numpy(np.zeros( ( rows-1, batch_size, K ) ) ) )
            
            #Compute weight of each path
            counter = 0
            for h in range( 0, HP ):
                weights_local[counter, :] = torch.mul( anc_weights[counter, :], torch.sigmoid( temper*torch.matmul( X.transpose( 0, 1 ), hp[:, h] ) ) )
                weights_local[counter+1, :] = torch.mul( anc_weights[ counter+1, :], torch.sigmoid( -temper*torch.matmul( X.transpose( 0, 1 ), hp[:, h] ) ) )
                counter += 2
            
            
            #Compute weighted sum of LDS
            for k in range( 0, K ):
                y_local[:, :, k] = torch.mul( weights_local[k, :], torch.matmul( p_LDS[:, :, k] + LD[:, :, k], X ) )
            
            y_pred = torch.sum( y_local, 2 ) 
            
            #Compute difference
            z = Y - y_pred
            
            #Compute MSE
            loss = torch.matmul( z, z.transpose( 0, 1 ) ).trace()/len(X[0,:])
            
            #Perform backprop
            loss.backward( )
            
            #Update parameters
            optimizer.step()

    return LD, hp



"""
Caculate gradients for  other nodes in the tree
"""
def calc_leaf_gradient( dim, y, x, LDS, nu, tau, leaf_path, K, depth, max_epoch, batch_size, LR, T ):
    #dim=dimension of latent space
    #y = output data
    #x = input data
    #LDS= linear dynamics of nodes in current depth of the tree
    #nu= hyperplanes
    #ancestor_weights= prpobability of previous paths
    #K=number of nodes at current depth of the tree
    #HP= number of hypeplanes
    #path_LDS= weighted sum of previous LDS
    #max_epoch = maximum number of epochs 
    #batch = size of batch
    # LR = learning rate
    #temper = parameter in sigmoid function used to make decision boundaries sharper
    
    nT = int( x[:, 0].size ) #Number of trajectories
    
    N = int( np.ceil( nT/ batch_size) )
    batch_size = int( batch_size)
    rows, cols = x.T.shape
    
    input_data = torch.from_numpy( x.T ).double() 
    output_data =  torch.from_numpy( y.T ).double()
    LD = Variable( torch.from_numpy( LDS ), requires_grad = True ).double()
    hp = Variable( torch.from_numpy( nu ), requires_grad = True ).double()
    tau = Variable( torch.from_numpy( tau ), requires_grad = True ).double()

    #Construct optimizer object
    optimizer = optim.SGD( [LD, hp], lr = LR,  momentum=0.95, dampening = 0, nesterov = True )
#    optimizer = optim.Adam([LD, hp], lr = LR)
    #Perform optimization
    for epoch in tqdm(range( max_epoch )):
        for n in range(N) :
            optimizer.zero_grad()
            
            if n == N-1:
                X = Variable( input_data[:, n*batch_size:] ).double()
                Y = Variable( output_data[:, n*batch_size:] ).double()
                weights_local = Variable( torch.from_numpy( np.zeros( ( K, len( input_data[0,n*batch_size:] ), depth - 1 ) ) ) ).double()
                y_local = Variable( torch.from_numpy(np.zeros( ( rows-1, len( input_data[0,n*batch_size:] ), K ) ) ) ).double()
            
            else:
                X = Variable( input_data[:, n*batch_size:(n+1)*batch_size] ).double()
                Y = Variable( output_data[:, n*batch_size:(n+1)*batch_size] ).double()
                weights_local = Variable( torch.from_numpy( np.ones( ( K, batch_size, depth - 1 ) ) ) ).double()
                y_local = Variable( torch.from_numpy(np.zeros( ( rows-1, batch_size, K ) ) ) ).double()
            
            #Compute weight of each LDS
            for k in range(K):
                for level in range(depth - 1):
                    idx = int(leaf_path[level, k])
                    nu_index = 2**level + idx-2 #Location of hyperplane in numpy array
                    child_idx = leaf_path[level + 1, k]
                    if np.isnan(child_idx) == False:
                        child_idx = int(child_idx)
                        # If odd then you went left
                        if child_idx % 2 == 1:
                            weights_local[k,:,level] = torch.sigmoid( torch.matmul( X.transpose( 0, 1 ), hp[:, nu_index] ) )
                        else:
                            weights_local[k,:,level] = torch.sigmoid( -1*torch.matmul( X.transpose( 0, 1 ), hp[:, nu_index] ) )
            
            #Take product along axis=2 to get tree structured stick breaking
            weights = torch.prod(weights_local, dim = 2)
            #Compute weighted sum of LDS
            for k in range( 0, K ):
                y_local[:, :, k] = torch.mul( weights[k, :], torch.matmul( LD[:, :, k], X ) ) - torch.mul( torch.exp(-tau*tau),X[:-1,:])
            
            y_pred = torch.sum( y_local, 2 ) 
            
            #Compute difference
            z = Y - y_pred
            
            #Compute MSE
            loss = torch.matmul( z, z.transpose( 0, 1 ) ).trace()/len(X[0,:])
            
            #Perform backprop
            loss.backward( )
            
            #Update parameters
            optimizer.step()

    return LD, hp