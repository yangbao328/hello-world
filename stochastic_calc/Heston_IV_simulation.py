#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:14:25 2026

@author: tianyang
"""
import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dynamic_hedging.the_engine import Options
from Heston_Implied_Vol import Heston_Volatility

#from the_engine import Options


T = 1; N = 252; rho = -0.0; kappa = 0.1; theta = 0.12
sigmav = 0.5; v0 = .3; r = 0.1; S0=100; K=90
heston = Heston_Volatility(T=T, N=N, rho=rho, kappa=kappa, theta=theta, 
                               sigma_v = sigmav, v0 = v0, r = r, S0 = S0)
strikes = np.arange(80, 135, 2)
ivols = []
calls = []
for K in strikes:
    
    ivols.append(heston.implied_vol(K, 0.05, 1, 30000))
    calls.append(heston.European_Call_MC_simu(K, 30000))

plt.plot(strikes, ivols, 'o-', markersize=3)
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.show()

#Volatility Skew and downward movement due to negative correlation rho
fig, ax1 = plt.subplots()
ax1.set_xlabel('Strike K')
ax1.set_ylabel('Implied Volatility', color='lightpink')
ax1.plot(strikes, ivols, color='lightpink', marker='o', markersize=2)
ax1.tick_params(axis='y', labelcolor='lightpink')

ax2 = ax1.twinx()
ax2.set_ylabel('Call, Heston', color='gray')
ax2.plot(strikes, calls, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

fig.tight_layout()
plt.title('Implied Volatility, individual MC-simulation at strike Ks')
plt.show()

#IV Smile and MC-simulation error reduced
iv_smile = heston.implied_vol_smile(strikes, 0.05, 1, 30000)

fig, ax1 = plt.subplots()
ax1.set_xlabel('Strike K')
ax1.set_ylabel('Implied Volatility', color='lightpink')
ax1.plot(strikes, iv_smile, color='lightpink', marker='o', markersize=2)
ax1.tick_params(axis='y', labelcolor='lightpink')

ax2 = ax1.twinx()
ax2.set_ylabel('Call, Heston', color='lightseagreen')
ax2.plot(strikes, calls, color='lightseagreen')
ax2.tick_params(axis='y', labelcolor='lightseagreen')

fig.tight_layout()
plt.title('Implied Volatility, MC-error reduced')
plt.show()

#print(f"Risk-neutral expectation for St is {S0 * np.exp(r*T):.3f}")


# Verify call price and implied volatility 
for K in [70, 100, 130]:
    hp = heston.European_Call_MC_simu(K, 50000)
    bp = Options(K=K, sigma = v0**0.5, r = r, T = T).eu_call_BS(S0, T-T)
    print(f"Strike at {K}, heston-call price {hp}, BS-call price {bp}")















