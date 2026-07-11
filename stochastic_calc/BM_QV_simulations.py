#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 18:24:16 2026

@author: tianyang
"""

import numpy as np
import matplotlib.pyplot as plt
from BM_QuadraticVariation import BrownianMotion

# 28June2026
bm = BrownianMotion(mu=0, sigma=1, T=1, N=252)  

t_inv, lnx_sim, lnx_vef = bm.ito_lnx_vects()     

max_error = np.max(np.abs(lnx_sim-lnx_vef))
print(f"Maximum error is {max_error:.2e}")
plt.plot(t_inv, lnx_sim, label='$ln_t^2$ direct')
plt.plot(t_inv, lnx_vef, '--', label='SDE (exact)')
plt.legend()
plt.show()   

# 24June2026

bm = BrownianMotion(mu=0, sigma=1, T=1, N=252)  
t_interval, W, W2_sim, W2_vef = bm.ito_verif_vector()  
max_error = np.max(np.abs(W2_sim-W2_vef))
print(f"Maximum error is {max_error:.2e}")
plt.plot(t_interval, W2_sim, label='$W_t^2$ direct')
plt.plot(t_interval, W2_vef, '--', label='SDE (exact)')
plt.legend()
plt.show()
#bm.BM_SDE(fdx=2x, fd2x=2)

bm = BrownianMotion(mu=0, sigma=1, T=1, N=252)  
t_interval, Xt_dir_geom, Xt_sde_geom = bm.ito_verif_vector_geom()  
max_error = np.max(np.abs(Xt_dir_geom-Xt_sde_geom))
print(f"Maximum error is {max_error:.2e}")
plt.plot(t_interval, Xt_dir_geom, label='$e^{-t}*x^3$ direct')
plt.plot(t_interval, Xt_sde_geom, '--', label='SDE (exact)')
plt.legend()
plt.show()

# Quadratic Variation
"""For a single Brownian path, compute of the sum of squared increments
   sum_{1, n} DeltaW^2 for n = 10, 100, 1000, 10000 steps all over the same total time T
   Show that over the mesh refines, the sum converges to T
   Provide reasonings for  dW^2 = dt
"""
steps_sample = [10, 100, 1000, 10000]
sum_steps = {}
for step in steps_sample:
    bm = BrownianMotion(mu=0, sigma=1, T=1, N=step)
    dW, W = bm.BM_simulation()
    sum_dW_sq = np.sum(dW**2)
    sum_steps[step] = sum_dW_sq
'''dX = dt = 1/sqrt(N) stems from scaling for Brownian Motion of time t, scaled and non-scaled Bt have x-axis as deltat and y-axis as Bt
   the scaled Brownian Motion is more stacked compared to spread-out non-scaled BM.
   W_j,deltat = Sj*deltaX 
   Assuming E[(Wt)^2] = 1 = E[(Sn*deltaX)^2] for reasonable scaling,
                      = Var((Sn^2)*deltaX^2) 
                      = deltaX^2*Var(Sn^2)
                      = deltaX^2*Var(X1+X2+...Xn)
                      = deltaX^2*(1*N) = 1 = N*deltaT
                      1/N = deltaX^2 = deltaT
'''
# Fundamental BM Simulation    
"""Create Python Class Brownian Motion that generates a path using the recursion
   W_{t+dt} = W_t + sqrt(dt)*Z, Z~N(0,1);
   Simulate one-year path with daily steps;
   Verify numerically E[W_t]=0, Var(W_t)=t using 10,000 paths. 
   Plot a few paths to visualize the "jitter"
"""
paths = 10000
simulations = np.zeros(paths)
bm = BrownianMotion(mu=0, sigma=1, T=1, N=252) 

for i in range(paths):
    #if to specify seed = 12 in this loop, all 10000 paths will follow the same sequence
    dW, W = bm.BM_simulation()
    simulations[i] = W[-1]
print(np.mean(simulations))
print(np.var(simulations))

plots = 6
plots_sim = {}
for i in range(plots):
    #if to specify seed = 12 inside loop, all simulated paths will follow the same sequence
    dW, simu = bm.BM_simulation()
    plt.plot(simu, label = f"Path {i+1}")

plt.title("BM Simulations")
plt.xlabel("Timestamps")
plt.ylabel("Random Walks")
plt.legend()    