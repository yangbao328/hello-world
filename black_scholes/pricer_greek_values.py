#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:52:26 2026

@author: tianyang
C(S,t) = e^(-r(T-t)) * (FN(d1) - KN(d2))
           = StN(d1) - Ke^(-r(T-t))N(d2)
         F = St*e^(r(T-t))
        d1 = log(F/K) / sigma*sqrt((T-t)) + sigma*sqrt(T-t)/2
        d2 = log(F/K) / sigma*sqrt((T-t)) - sigma*sqRt(T-2)/2
        
dSt = 0 St dt + -sigma^2/2 St dWt
lnSt has a negative drift of -sigma^2/2 
This is not a contradiction – it’s a consequence of the convexity of the exponential. 
Over time, the distribution of St becomes skewed: 
    the mean stays at S0 (zero drift), 
    but the median falls to S0 e^(-sigma^2*T/2)
    and the mode is even lower at S0 e^(-3sigma^2*T/2).
The distribution is pushed out to the right by occasional large gains, but most paths end below S0.       
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
        
    def d1_t0(self):
        
        #t = np.linspace(0, self.T, self.n_steps)
        
        d1 = (np.log(self.S0/self.K) + (self.r + self.sigma**2/2 )*self.T )/ (self.sigma*np.sqrt(self.T))
        #d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        return d1
    

    def BS_price_t0(self):
        
        #t = np.linspace(0, self.T, self.n_steps)
        
        #F = self.S0*np.exp(self.r*(self.T-t)) - erroneous
        #Forward price should be priced with time-series St instead of single S0
        #the idea is St moves over time instead of a static number from 0 to t to T
    
        d1 = self.d1_t0()
        Nd1 = norm.cdf(d1)
        #d2 = np.log(self.S0/self.K) / (self.sigma*np.sqrt(self.T-t)) - 0.5*(self.sigma*np.sqrt(self.T-t))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        Nd2 = norm.cdf(d2)

        C = self.S0*Nd1 - np.exp(-self.r*self.T)*self.K*Nd2
        
        #C = np.exp(-self.r*(self.T-t))*(F*Nd1 - self.K*Nd2)
        #need proper time-series St to calculate meaningful Forward price for time-series C
        
        return C
    
    def St_simu(self, n_paths):
        
        '''
        St = S0 e^((r-sigma^2/2)T + sigma*sqrt(T)Z)
        '''
        Z = self.rng.normal(0, 1, size=n_paths)
        St = self.S0 * np.exp((self.r-self.sigma**2/2)*self.T + self.sigma*(self.T**0.5)*Z)
        
        return St

    def MC_price(self, n_paths):
        
        St = self.St_simu(n_paths)
        
        payoff = np.maximum(St-self.K,0) 
        #np.maximum(St-self.K, 0) produces element-wise payoff
        #np.max(St-self.K, 0) produces single max among entries in the array
        
        discounted = np.exp(-self.r*self.T)*payoff
        
        po_PV_mean = np.exp(-self.r*self.T)*np.mean(payoff)
        #Monte-Carlo simulation, average of discounted payoff according to European Call
        
        return discounted, po_PV_mean
    
    def Nd1Prime(self):
        '''
        N'(x) = 1/sqrt(2pi) * e^((-x^2)/2)
        '''
        
        d1 = self.d1_t0()
        
        return 1/np.sqrt(np.pi*2) * np.exp((-d1**2)/2)
    
    def Delta(self):
        
        d1 = self.d1_t0()
        
        return norm.cdf(d1)
    
    def Gamma(self):
        
        Np = self.Nd1Prime()
        
        return Np/(self.S0*self.sigma*np.sqrt(self.T))
    
    def Theta(self):
        
        d1 = self.d1_t0()
        d2 = d1 - self.sigma*np.sqrt(self.T)
        Np = self.Nd1Prime()
        
        term1 = -self.S0*Np*self.sigma/(np.sqrt(self.T)*2)
        term2 = -self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        
        return term1 + term2
    
    def Vega(self):
        
        d1 = self.d1_t0()
        
        return self.S0 * norm.cdf(d1) * np.sqrt(self.T)

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


axes[0].plot(S0s, delta, label='Delta')
axes[0].legend()
axes[0].set_ylabel('Delta')


axes[1].plot(S0s, gamma, label='Gamma')
axes[1].legend()
axes[1].set_ylabel('Gamma')
axes[2].plot(S0s, theta, label='Theta')
axes[2].legend()
axes[2].set_ylabel('Theta')
axes[3].plot(S0s, vega, label='Vega')
axes[3].legend()
axes[3].set_ylabel('Vega')

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

bs = BlackScholes()
Sts = bs.St_simu(paths)
#St_K = [1 for St in Sts if St > bs.K]
St_K = np.mean(Sts > bs.K)

d2 = bs.d1_t0() - bs.sigma*(bs.T**0.5)

print(St_K, norm.cdf(d2))
