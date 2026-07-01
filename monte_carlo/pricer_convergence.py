#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:47:25 2026

@author: tianyang

# 27June 10:37AM - 10:55AM

Build a Monte-Carlo Pricer for European Call;
Show convergence and 1/N^0.5 error scaling.
    a.1/sqrt(N) is per Central Limit Theorem via MC sample mean to find option price
    b. Large number of observation with the same distribution is made, 
    as C = sum_{1}^{n} (discounted payoff)) / number of observations
    c. discounted = np.exp(-self.r*self.T)*payoff
    c.1. MC_price = np.mean(discounted)

By CLT, distribution C is approximately normal with mean Ctrue and SE sigma/N**0.5

log(SE) = log(sigma/N**0.5) = log(sigma) - 1/2 log(N)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy 
from scipy.stats import norm


class BlackScholes:
    def __init__(self, S0=100, K=100, T=1, r=0.05, sigma=0.2, n_steps=252, seed=None):
        
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        self.rng = np.random.default_rng(seed)
    
    def MC_price(self, n_paths):
        
        St = self.St_simu(n_paths)
        
        payoff = np.maximum(St-self.K,0) 
        #np.maximum(St-self.K, 0) produces element-wise payoff
        #np.max(St-self.K, 0) produces single max among entries in the array
        
        discounted = np.exp(-self.r*self.T)*payoff
        
        po_PV_mean = np.exp(-self.r*self.T)*np.mean(payoff)
        #Monte-Carlo simulation, average of discounted payoff according to European Call
        
        return discounted, po_PV_mean

paths = 100000
bs = BlackScholes()
price_BS = bs.BS_price_t0()

discounted, price_MC = bs.MC_price(paths)
#discounted payoff

MC_simu = np.mean(discounted) 
#MC-simulated option price

running_mean = np.cumsum(discounted)/np.arange(1,paths+1)

running_varc = np.cumsum((discounted - running_mean)**2)/np.arange(1,paths+1)
running_stdv = running_varc**0.5
standard_err =  running_stdv/np.arange(1, paths+1)**0.5 # 1/N^0.5

plt.plot(running_mean, label='MC Running Average')
plt.axhline(price_BS, color='pink', linestyle='--', label=f'BS Price {price_BS:.4f}')
plt.xlabel('Number of Paths')
plt.ylabel('Option Price')
plt.title('Monte Carlo Convergence of European Call Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(bs.BS_price_t0() - 0.5, bs.BS_price_t0() + 0.5)  # zoom in
plt.fill_between(range(paths), running_mean - 1.96*standard_err, running_mean + 1.96*standard_err,
                 alpha=0.3, color='blue', label='95% Confidence Interval')

plt.show()