#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 19:18:37 2026

@author: tianyang

Heston Model, stochastic volatility
a. St and vt, 13 July
b. Monte-Carlo multi-path European call pricer, 14 July
c. Implied volatility and plots
"""

""""
UNDER Q-probability measurement
dSt = mu*St*dt + vt**0.5*St*dWt^S
dvt = k(theta-vt)dt + sigmav*vt**0.5*dWt^v

Parameters
 kappa: rate at which volatility mean-reverts to its long-term average
 theta: long-run average variance of asset
sigmav: volatility of variance process
   rho: correlation coefficient between the asset price shocks and volatility shocks
    v0: initial variance of asset
    
dWt^S and dWt^v brownian simulation   --- does dWt always represent brownian motion, standard?
kappa, theta sigmav are derived from vt; how to define vt?
rho: links St-shock and vt-shock, since shock then delta values, dWt^S and dWt^v; why not dSt, dvt
     cholesky factor to produce correlated series  
"""
import numpy as np

T = 1; N = 252
mu_dWtS = 0; sigma_dWts = (T/N)**0.5 ; # sigma_dWt (deltat)**0.5

rho = -0.7
rng = np.random.default_rng()

S0=100; v0 = .3

kappa = 0.1; theta = 0.12; sigmav = 0.5; r = 0.1

#vt+i = vti + kappa(theta - vti)*deltat + sigmav (vti)**0.5 *dWtv; max(vt,0)
#St+i = Sti + rSti*deltat + vti**0.5 Sti dWtS
dt = T/N

St = np.zeros(N+1); Vt = np.zeros(N+1)
St[0] = S0; Vt[0] = v0
n_p = 10000
ST = np.zeros(n_p)
for each in range(n_p):             #verify Heston SDE simulation
    
    St = np.zeros(N+1); 
    Vt = np.zeros(N+1)
    St[0] = S0; Vt[0] = v0
    
    for i in range(N):
        Z1 = rng.normal(0, 1)       #N~(0,1)
        Z2 = rng.normal(0, 1)
        dWtS = (dt)**0.5*Z1         #N~(0,deltat**0.5)
        dWtv = (dt)**0.5*(rho*Z1+(1-rho**2)**0.5*Z2)
        Vt[i+1] = np.maximum(Vt[i] + kappa * (theta - Vt[i]) * dt + sigmav * Vt[i]**0.5 * dWtv,0)
        St[i+1] = St[i] + r * St[i] * dt + Vt[i]**0.5 * St[i] * dWtS
    ST[each] = St[-1]
print("Mean S_T:", np.mean(ST)) 


#%%
#vt+i = vti + kappa(theta - vti)*deltat + sigmav (vti)**0.5 *dWtv; max(vt,0)
#St+i = Sti + rSti*deltat + vti**0.5 Sti dWtS

Z1 = np.random.normal(mu_dWtS, 1, N)
Z2 = np.random.normal(mu_dWtS, 1, N)

dWtS = (dt)**0.5*Z1
dWtv = (dt)**0.5*(rho*Z1+(1-rho**2)**0.5*Z2)

St = np.zeros(N+1); Vt = np.zeros(N+1)
St[0] = S0; Vt[0] = v0

for i in range(N):
    
    #volatility has floor 0, as being standard deviation**2
    Vt[i+1] = max(Vt[i] + kappa * (theta - Vt[i]) * dt + sigmav * Vt[i]**0.5 * dWtv[i], 0)
    St[i+1] = St[i] + r * St[i] * dt + Vt[i]**0.5 * St[i] * dWtS[i]

print(St[-1])













#NOTES and First Attempt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

corr = 0.9
vdWtS = sigma_dWts**2       #INCORRECT, dWt standard deviation == deltat**0.5
vdWtv = 0.6                 #INCORRECT, dWt standard deviation == deltat**0.5
#covrn = vdWtS*vdWtv*corr
covMx = [[vdWtS**2, vdWtS*vdWtv*corr],[vdWtS*vdWtv*corr, vdWtv**2]]    #INCORRECT, design correlation matrix with rho
L_chk = np.linalg.cholesky(covMx)

#random shocks to dWtS and dWtv are not introduced and dWtS dimension is off
dWtS = np.random.normal(loc=mu_dWtS, scale=sigma_dWts, size=(len(covMx),N)) 
dWtv = L_chk@dWtS
S0 = 100; mu_St = 0; sigma_St = 1.0; 
#terminal St is not appropraite; 
#intermediate St is needed to incorporate Heston SDE, with volatility at each step

#vt+i = vti + kappa(theta - vti)*deltat + sigmav (vti)**0.5 *dWtv; max(vt,0)
#St+i = Sti + rSti*deltat + vti**0.5 Sti dWtS

St= 100 * np.exp(mu_St - 0.5 * sigma_St**2) * T + sigma_St*(T**0.5)*dWtS[0]








