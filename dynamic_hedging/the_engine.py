#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 19:06:27 2026

@author: tianyang

This dynamic hedging engine simulates 
stock movement, options and Greeks evolvement, order execution and P&L monitoring
enabled by Market, Options and Hedger, the three fundamental classes. 

Goals
simulate cost of continuously delta‑hedging a short option equals the Black‑Scholes price,

Convey final P&L converges to zero (minus transaction costs) when the stock follows the risk‑neutral process.

"""

import numpy as np
#import matplotlib.pyplot as plt
#from scipy.stats import norm
class Market:
    '''
    Goal
    Simulate stock price via Geometric Brownian Motion
    
    Note
    Log-transform GBM from exponential format to implement in vectorised additive manner
    '''
    
    
    def __init__(self, S0, r, sigma, T, N, seed=None):
        '''
        mu referenced in St simulation follows risk-neutral measurement
        '''
        self.S0 = S0
        self.T = T
        self.sigma = sigma
        self.mu = r
        self.N = N
        self.rng = np.random.default_rng(seed)#mean, sigma, size

    def BM_Bt(self):
        '''
        Underlying dBt term in Brownian Motion, Bt = sum_{i=0}^{n} Xi
        referenced in St diffusion,
        dWt^2 = dt per Quadratic Variation
        sigma=(T/N)^0.5 scaling factor for Var(Bt)=T and per Qudratic Variation
        '''
        
        dBt = self.rng.normal(0, (self.T/self.N)**0.5, self.N) #CORRECTION mean=0
        Bt = np.zeros(self.N+1)
        Bt[1:] = np.cumsum(dBt)
        
        return dBt, Bt
    
    def GBM_St(self):
        '''
        Y = lnX
        St = X = e^Y
        
        dXt = mu * Xt * dt + sigma * Xt * dBt      --- underlying dXt via ito's lemma, multiplicative Xt following GBM
        dY = (mu - .5*sigma**2) dt + sigma dBt     --- independent dY from Y_n-1
        '''
        
        dBt, Bt = self.BM_Bt()
        dt = self.T/self.N
        
        drift_Y = (self.mu - 0.5*self.sigma**2) * dt
        diffn_Y = self.sigma * dBt
        
        Y = np.zeros(self.N+1)
        Y[0] = np.log(self.S0)
        Y[1:] = np.log(self.S0) + np.cumsum(drift_Y+diffn_Y) # additive BM observed from dY that relies on initial dt and dBt
        
        return np.exp(Y)
    
    def GBM_St_terminal(self):
        
        Z = self.rng.normal(0, 1)
        St = self.S0 * np.exp((self.mu - 0.5 * self.sigma**2) * self.T
                             + self.sigma * np.sqrt(self.T) * Z)
        return St

market = Market(S0 = 100, mu = 0, sigma = 1, T = 1, N = 252, seed = 12)
print(market.GBM_St()[-1])
        
class Optins:
    
    '''
    Goal
    Simulate current price of the option
    Illustrate current delta (and other Greeks)

    Notes
    Following risk-neutral measurement Q, Black-Scholes assumes drift-term of St is upon r
    Stock simulation in class Market shares risk-free rate r for St drift
    '''
    
    def __init__(self):
        pass
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        