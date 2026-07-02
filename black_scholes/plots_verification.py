#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 20:19:08 2026

@author: tianyang
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from bs_greek import BlackScholes
paths = 10000
S0 = 100

bs = BlackScholes(S0=S0)
price_BS = bs.BS_price_t0()
discounted, price_MC = bs.MC_price(paths)

# Greek Values  

delta = []
gamma = []
theta = []
vega = []
greeks = {'delta':delta, 'gamma':gamma, 'theta':theta, 'vega':vega}

S0s = range(50,151,5)
for S0 in S0s:
    
    bs = BlackScholes(S0=S0)
    delta.append(bs.Delta())
    gamma.append(bs.Gamma())
    theta.append(bs.Theta())
    vega.append(bs.Vega())

fig, axes = plt.subplots(4,1,figsize = (3,2),sharex=True)

for i in range(len(greeks.keys())):
    
    greek = list(greeks.keys())[i]
    
    axes[i].plot(S0s, greeks[greek], label=greek)
    axes[i].legend()
    axes[i].set_ylabel('Delta')

print(f'Strike at {bs.K}')

# 27June 10:37AM - 10:55AM
'''
To verify N(d2)(risk‑neutral probability of expiring ITM):

Simulate 100,000 paths of St 
under Q: St = S0 * e^((r-sigma^2/2)T + sigma*sqrt(T)Z) with Z∼N(0,1).

Count the fraction of paths where St>K and the fraction should equal to N(d2)
Compare the simulated fraction with norm.cdf(d2).
'''

paths = 100000

Sts = bs.St_simu(paths)
#St_K = [1 for St in Sts if St > bs.K]
St_K = np.mean(Sts > bs.K)

d2 = bs.d1_t0() - bs.sigma*(bs.T**0.5)

print(St_K, norm.cdf(d2))
