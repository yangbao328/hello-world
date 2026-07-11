#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:57:07 2026

@author: tianyang

# Brownian Motion 24June, Wed.
Ito's Lemma Implementation and verifying against direct calculation for f(t,x)

# Quadratic Variation
For a single Brownian path, compute of the sum of squared increments
   sum_{1, n} DeltaW^2 for n = 10, 100, 1000, 10000 steps all over the same total time T
   Show that over the mesh refines, the sum converges to T
   Provide reasonings for  dW^2 = dt

EXAMPLES
1. f(x) = x^2 
ADDITIVE Brownian Motion where f(x) = f(x0) + ... 
f(x) increment, df, following ito's formula
f(x) to be sum_{1}^{n} f(x) increment for each step
with N-step increment, pay attention to f(x0) as the first step

2. f(t,x) = e^(-t)*x^3
MULTIPLICATIVE Brownian Motion where f(x) = f(x0) * ...; Geometric BM
X at each step, with dXt = mu * Xt * dt + sigma * Xt *dBt, per GBM
   drift, dt-term = (self.mean - self.var/2) * t
   diffusion, dBt-term = self.var**0.5 * Bt
   Xt= Xt0 * np.exp(X_dft+X_fsn)
   i.e. Xt[i+1] = Xt[i] + self.mean * self.dt + self.var**0.5 * Xt[i] * dBt[i]

f(t,x) at each step, with analytical df(t,x)
   df(t,x) following ito's formula; in layout of dt and dSt term
   with N-step increment, f(x0) != 0, otherwise f(t,x) remains 0 as well since multiplicative of GBM

dXt has shape (1,N) while f(t,x) has shape (1,N+1), because f(t,x) has N interval
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy 
from scipy.stats import norm

class BrownianMotion:
    
    def __init__(self, mu, sigma, T=1, N=252, seed=None):
        self.mean = mu
        self.sigma = sigma
        self.var = 1
        self.T = T
        self.N = N
        self.dt = self.T/self.N
        self.rng = np.random.default_rng(seed)
        '''having seed reset for every simulation offers statistically independent path'''
        

    def BM_simulation(self,):
        
        #dWt = np.sqrt(self.dt)*self.rng.normal(0, 1, self.N)
        dWt = self.rng.normal(0,np.sqrt(self.T/self.N), self.N)
        #to scale dWt so that variance of Wt is linear to time T
        
        #Var(deltaW) = deltat*(sigma^2) => Var(W) = n * (deltat) * (sigma^2) = n * (T/N) * (sigma^2) = T * sigma^2
        
        W = np.zeros(self.N+1)
        W[1:] = np.cumsum(dWt)
        
        return dWt, W
 
    def ito_verif_vector(self):
        
        dWt, W_dir = self.BM_simulation()
        W2_dir = W_dir**2
        
        W2_sde = np.zeros(self.N+1)

        W2_inc = 2 * W_dir[:-1] * dWt + dWt**2
        #calculate increment only, from underlying Wt, dWt and t
        #W_dir[:-1] excludes nth Wt because the one last item happens from Wt-1 to Wt
        #with Wt-1, the total increment between n-1 and nth step is 2*Wn-1*dWn-1+(dWn-1)^2
        #W2_inc = W2_sde[:-1] + 2 * W_dir * dWt + dWt**2
        
        W2_sde[1:] = np.cumsum(W2_inc)
        
        t = np.linspace(0, self.T, self.N+1)
        #to include initial 0, and 252 following steps
        #t = np.linspace(0, self.T, self.N)

        return t, W_dir, W2_dir, W2_sde
    
    def ito_lnx_vects(self):
        '''
        f(x) = lnX
        dXt = mu * Xt * dt + sigma * Xt * dBt      --- underlying dXt via ito's lemma, multiplicative Xt following GBM
        dfx = (mu - .5*sigma**2) dt + sigma dBt    --- drift and diffusion not dependent on Xt, hence additive implementation
        '''
        dBt, Bt = self.BM_simulation()
        
        t = np.linspace(0, self.T, self.N+1)
        sigma = self.var**0.5
        X0 = 1
        
        lnx_sde = np.log(X0) + (self.mean - 0.5*self.var) * t + sigma * Bt
        #lnx_inc = (self.mean - 0.5*self.var) * t + sigma * Bt --- increment needs dt and dBt instead of t and Bt [ERROR]
        #lnx_sde = np.zeros(self.N+1)
        #lnx_sde[1:] = np.cumsum(lnx_inc) --- ADDITIVE to find f(x) because drift and diffusion term of df(x) does NOT depend on Xt, while Xt~GBM
        
        X_drift = (self.mean - 0.5*sigma**2) * t
        X_diffn = sigma * Bt

        Xt = X0*np.exp(X_drift+X_diffn) #GBM Xt
        lnx_dir = np.log(Xt)
        
        return t, lnx_dir, lnx_sde
    
    def ito_verif_vector_geom(self):
        '''
        f(t,x) = e^(-t)*x^3
        dXt = mu * Xt * dt + sigma * Xt *dBt
        df(t,x) = (-1+3m+3sigma^2)*f(t,x)*dt+3sigma*f(t,x)dBt
        
        Xt = X0 * e^((mu - sigma**2/2)*t + sigma*Bt) = X0 * e^(drift + diffusion)
        '''
        dBt, Bt = self.BM_simulation() #N, N+1    
        
        t = np.linspace(0, self.T, self.N+1)

        #Xt = np.zeros(self.N+1)
        Xt0 = 1
        X_dft = (self.mean - self.var/2) * t
        X_fsn = self.var**0.5 * Bt
        Xt= Xt0 * np.exp(X_dft+X_fsn)
    
        f_dir = np.exp(-t)*Xt**3
        
        #f_tx = np.zeros(self.N+1)
        f_tx0 = np.exp(0) * Xt[0]**3
        
        alpha = -1 + 3*self.mean + 3*self.var
        beta = 3*self.var**0.5
        f_dft = (alpha - beta**2/2) * t
        f_fsn = beta * Bt
        f_tx = f_tx0 * np.exp(f_dft+f_fsn)
        
        return t, f_dir, f_tx
        
    def ito_verification(self,):
        #bm = BrownianMotion()
        dWt, W_dir = self.BM_simulation()
        W2_dir = W_dir**2
        
        W2_sde = np.zeros(self.N+1)
        for i in range(self.N):
            W2_sde[i+1] = W2_sde[i] + 2 * W_dir[i] * dWt[i] + dWt[i]**2 #self.dt
        
        t = np.linspace(0, self.T, self.N+1)
        
        return t, W_dir, W2_dir, W2_sde
    
    def ito_verification_geom(self):
        '''
        f(t,x) = e^(-t)*x^3
        dXt = mu * Xt * dt + sigma * Xt *dBt
        df(t,x) = (-1+3m+3sigma^2)*f(t,x)*dt+3sigma*f(t,x)dBt
        '''
        
        dBt, _ = self.BM_simulation()
        Xt = np.zeros(self.N+1)
        Xt[0] = 1
        #initial value for Xt not equal to 0, o.w. it stays at 0
        for i in range(self.N):
            Xt[i+1] = Xt[i] + self.mean * self.dt + self.var**0.5 * Xt[i] * dBt[i]

        t = np.linspace(0, self.T, self.N+1)
        f_dir = np.exp(-t)*Xt**3   
         
        f_tx = np.zeros(self.N+1)
        f_tx[0] = np.exp(-0)*Xt[0]**3
        #f_tx[0] is initiated with Xt[0]
        
        for i in range(self.N):
            f_tx[i+1] = f_tx[i] + (-1 + 3 * self.mean + 3 * self.var) * f_tx[i] * self.dt + 3 * self.var**0.5 * f_tx[i] * dBt[i]
            
        return t, f_dir, f_tx

