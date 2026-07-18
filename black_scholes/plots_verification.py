#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 20:19:08 2026

@author: tianyang
"""
import sys, os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
from Heston_Implied_Vol import Heston_Volatility
from dynamic_hedging import Options

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

#European Call via Heston Model
#Implied volatility plots
T = 1; N = 252; rho = -0.0; kappa = 0.1; theta = 0.12
sigmav = 0.5; v0 = .3; r = 0.1; S0=100; K=90; n_paths = 30000
sigma_low = 0.05; sigma_high = 1
strikes = np.arange(80, 135, 2)

heston = Heston_Volatility(T=T, N=N, rho=rho, kappa=kappa, theta=theta, 
                               sigma_v = sigmav, v0 = v0, r = r, S0 = S0)

calls = [heston.European_Call_MC_simu(K, n_paths) for K in strikes]
iv_smiles = heston.implied_vol_smile(strikes, sigma_low, sigma_high, n_paths)

fig, ax1 = plt.subplots()
ax1.set_xlabel('Strike K')
ax1.set_ylabel('Implied Volatility', color='lightpink')
ax1.plot(strikes, iv_smiles, color='lightpink', marker='o', markersize=2)
ax1.tick_params(axis='y', labelcolor='lightpink')

ax2 = ax1.twinx()
ax2.set_ylabel('Call, Heston', color='lightseagreen')
ax2.plot(strikes, calls, color='lightseagreen')
ax2.tick_params(axis='y', labelcolor='lightseagreen')

fig.tight_layout()
plt.title('Implied Volatility, MC-error reduced')
plt.show()
