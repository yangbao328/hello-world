#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:40:56 2026

@author: tianyang

Cholesky Project 13June2026 13:45 - 15:38

   Create covariance matrix for two assets. Decompose the covariance matrix.
   Generate 10,000 paths of 2 correlated random walks. 
   Plot 50 of them to visually inspect correlation.
   Calculate the correlation matrix of increment-level simulation (xxxfinal price levels to verifyxxx) it matches the output.
   
THE CORRECTION is on checking correlation increment-level simulation with initial covariance matrix
is because a. Choleski factor is multiplied with simulated normal random series, the concept in BM corresponds to increment-level as return in finance
           b. Price-level series is obtained by np.cumsum and it leads the series to be un-stationary with spruious/random correlation of two variables/dimensions
"""

import numpy as np
import matplotlib.pyplot as plt

#to decompose a covariance matrix, 
#generate Choleski Triangular Matrix that functions as coefficient, (Choleski factor as L, linear transformation)
#multiply generated matrix with a (vector of independent normal random numerates ) variable 
#to produce correlated (correlated random) variable +as specified by the initial covariance matrix


#Define asset 1 and 2 volatility, correlation of asset 1 and 2, and variance-covariance matrix
#diagonal of the matrix is varaince, off-diagonal is covariance
#covariance = sigma1 * sigma2 * correlation


v1 = 0.3 
v2= 0.6 #vol to variance
corr = 0.9
covr = v1*v2*corr
covM = [[v1*v1, covr ],[covr, v2*v2]]
L = np.linalg.cholesky(covM); L@L.T==covM; np.allclose(L@L.T, covM)


dimensions = len(covM)
sample_path_CKY = 10000
steps = 252
uncorr_spl_3d = np.random.normal(size=(dimensions, sample_path_CKY, steps)) 
uncorr_spl_2d = uncorr_spl_3d.reshape(dimensions, -1)          # (2, 10000*252)

corred_spl_2d = L@uncorr_spl_2d #increment deltaW in BM, change in the process
corred_spl_3d = corred_spl_2d.reshape(dimensions, sample_path_CKY, steps)

random_walk_2d = np.cumsum(corred_spl_2d,axis=1) # SUM(deltaW), total change, example as arithmetic sum
random_walk_3d = np.cumsum(corred_spl_3d,axis=2) # axis=2 to sum across timesteps

#axis=1 across the column to arrive at sum for the row
#Geometric BM represents price level of stock from changes over time 

'''VERIFY Correlations
   a. correlation matrix from volatility and covariance defined
   b. correlation matrix from correlated simulation of two variables
   c. correlation matrix from increments of random walk generated
'''

std_devs = np.sqrt(np.diag(covM))
d_matrix_inv = np.diag(1 / std_devs)
corr_matrix = d_matrix_inv @ covM @ d_matrix_inv
np.corrcoef(random_walk_2d) == corr_matrix; np.allclose(np.corrcoef(random_walk_2d), corr_matrix)
#return FALSE because RandomWalk, price level, are not stationary per np.cumsum, which leads to spurious correlation
#summation makes the series un-stationary?
#CORRECTION UNDERSTANDING: np.cumsum is arithmetic BM with drift = 0

np.corrcoef(corred_spl_2d) == corr_matrix; np.allclose(np.corrcoef(corred_spl_2d), corr_matrix)
#CLOSE ENOUGH because the Choleski factor is applied to define return level information

increment = np.diff(random_walk_2d, axis = 1) #axis=1 to take diff across column and produce result of the row dimension
np.allclose(np.corrcoef(increment), corr_matrix)

plt.figure()
plt.plot(corred_spl_2d[0,:50], label='Asset 1 increments')
plt.plot(corred_spl_2d[1,:50], label='Asset 2 increments')
plt.legend()
plt.show()

plt.figure()
plt.plot(random_walk_2d[0,:50], label='Asset 1 Price')
plt.plot(random_walk_2d[1,:50], label='Asset 2 Price')
plt.plot()
plt.show()


fig, ax = plt.subplots(figsize=(3, 2))
 
for i in range(10):
    ax.plot(random_walk_3d[0, i, :], alpha = 0.7, label=f'Path{i}' if i==0 else "")
    ax.plot(random_walk_3d[1, i, :], alpha = 0.7, linestyle='--')
ax.set_title('10 Correlated Random Walk Paths (Asset 1 solid, Asset 2 dashed')

plt.figure(figsize=(3,2))
plt.scatter(corred_spl_2d[0, :1000], corred_spl_2d[1, :1000], alpha=0.3, s=.2)
plt.xlabel('Asset 1 increment')
plt.ylabel('Asset 2 increment')
plt.title('Scatter of correlated increments (first 1000 points)')
plt.axis('equal')
plt.show()