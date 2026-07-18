#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 19:18:37 2026

@author: tianyang

Heston Model, stochastic volatility
a. St and vt, 13 July
b. Monte-Carlo multi-path European call pricer, 14 July
c. Implied volatility smile by inverting B-S for different strikes
"""
import sys
import os
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dynamic_hedging'))
from the_engine import Options

class Heston_Volatility:
    
    '''
    UNDER Q-probability measurement
    dSt = mu*St*dt + vt**0.5*St*dWt^S
    dvt = k(theta-vt)dt + sigmav*vt**0.5*dWt^v
   vt+i = vti + kappa(theta - vti)*deltat + sigmav (vti)**0.5 *dWtv; max(vt,0)
   St+i = Sti + rSti*deltat + vti**0.5 Sti dWtS
   
    Parameters
     kappa: rate at which volatility mean-reverts to its long-term average
     theta: long-run average variance of asset
    sigmav: volatility of variance process
       rho: correlation coefficient between the asset price shocks and volatility shocks
        v0: initial variance of asset
    '''
    

    def __init__(self, T, N, rho, kappa, theta, sigma_v, v0, r, S0):
        
        self.T = T
        self.N = N
        self.rho = rho
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.v0 = v0
        self.r = r
        self.S0 = S0
        
        self.dt = self.T/self.N
        self.rng = np.random.default_rng()
    
    def St_heston_simu(self, n_paths):
        '''
        Vectorise n-path simulations for St over N timestamps
        Maintain time-wise loop for St and Vt as dependent on St-1 and Vt-1 respectively
        '''
        
        S = np.full(n_paths, self.S0, dtype=float)
        v = np.full(n_paths, self.v0, dtype=float)
        
        Z1 = self.rng.normal(0, 1, size = (n_paths, self.N))
        Z2 = self.rng.normal(0, 1, size = (n_paths, self.N))
    
        dWtS = self.dt**0.5*Z1                                          #array of N~(0,1)
        dWtv = self.dt**0.5 * (self.rho*Z1+(1-self.rho**2)**0.5*Z2)     #delta volatility, correlated with delta S
        
        for t in range(self.N):
            
            v_left = v                                                  #v at one-previous step, update once St is calculated
            v = np.maximum(v + self.kappa*(self.theta - v)*self.dt + self.sigma_v*np.maximum(v,0)**0.5*dWtv[:, t], 0)
            
            S = S + self.r*S*self.dt + v_left**0.5*S*dWtS[:,t]          #simualte S using v_t-1 to maintain correlation
            
        return S 
    
    def European_Call_MC_simu(self, K, n_paths):
        
        S_n_paths = self.St_heston_simu(n_paths)
        payoff_np = np.maximum(S_n_paths - K, 0)
        Call_MC = np.exp(-self.r*self.T)*np.mean(payoff_np)
        
        return Call_MC
    
    
    def implied_vol(self, K, sigma_low, sigma_high, n_paths):
        '''
        C_BS = S0N(d1) - e^(-rT) KN(d2)
        Find sigma_BS such that C_BS = C_heston given S0, K, T and r
        '''
        r=self.r; T=self.T; S0=self.S0; T0=T-T
    
        #function below and ST_simulator introduces MC errors as
        #new Z1 Z2 are generated for each strike K
        call_heston = self.European_Call_MC_simu(K, n_paths)
        #print(f"Call, Heston:{call_heston:.3f}, \
        #        Call, sigma lower bound:{call_BS_low:.3f}, \
        #        Call, sigma higher bound: {call_BS_high:.3f}")
        
        f = lambda sigma: Options(K=K, sigma=sigma, r=r, T=T).eu_call_BS(S0, T0) - call_heston
        
        return brentq(f, sigma_low, sigma_high)
    
    def implied_vol_smile(self, Ks, sigma_low, sigma_high, n_paths):
        '''
        Generate St and share it with MC-Call simulation across Ks
        to reduce Monte-Carlo uncorrelated erros
        '''        
        r=self.r; T=self.T; S0=self.S0; T0=T-T
        St = self.St_heston_simu(n_paths)
        
        ivol = []
        
        for K in Ks:
            
            payoff = np.maximum(St-K,0)
            call_heston = np.exp(-r*T)*np.mean(payoff)
            f = lambda sigma: Options(K=K, sigma=sigma, r=r, T=T).eu_call_BS(S0, T0) - call_heston
            iv = brentq(f, sigma_low, sigma_high)
            
            ivol.append(iv)
        
        return ivol
    
    def St_heston(self):
        '''
        One-path stock price simulation, N timestamps
        '''
        
        N = self.N
        St = np.zeros(N+1); St[0] = self.S0
        Vt = np.zeros(N+1); Vt[0] = self.v0
        
        for i in range(self.N):
            Z1 = self.rng.normal(0, 1)              #N~(0,1)
            Z2 = self.rng.normal(0, 1)              #CORRECTION from N~(0, deltat**0.5); 
                                                    #error causes dWt~N(0, dt**2) with significantly smaller stdev
            dWtS = (self.dt)**0.5*Z1                #N~(0,deltat**0.5)
            dWtv = (self.dt)**0.5*(self.rho*Z1+(1-self.rho**2)**0.5*Z2)
            Vt[i+1] = np.maximum(Vt[i] + self.kappa * (self.theta - Vt[i]) * self.dt + self.sigma_v * np.maximum(Vt[i]**0.5,0) * dWtv,0)
            St[i+1] = St[i] + self.r * St[i] * self.dt + Vt[i]**0.5 * St[i] * dWtS
        
        return St
    
    def European_Call_MC(self, K, n_paths):
        
        St_MC = np.zeros(n_paths)
        payoff = np.zeros(n_paths)
        
        for n in range(n_paths):
            St = self.St_heston()[-1]
            call_payoff = np.maximum(St - K,0)
            
            St_MC[n] = St
            payoff[n] = call_payoff

            #print(St, call_payoff)
        call_MC = np.exp(-self.r*self.T) * np.mean(payoff)
        
        return call_MC

if __name__ == '__main__':
    T = 1; N = 252; rho = -0.7; kappa = 0.02; theta = 0.12
    sigmav = 0.5; v0 = .3; r = 0.1; S0=100; K=90

    heston = Heston_Volatility(T=T, N=N, rho=rho, kappa=kappa, theta=theta, 
                               sigma_v = sigmav, v0 = v0, r = r, S0 = S0)
    options = Options(K=K, sigma = v0**0.5, r = r, T = T)

    european_C = heston.European_Call_MC(90, 1000)
    eu_MC = heston.European_Call_MC_simu(90, 1000)
    print(f"Single-path European Call:{np.mean(heston.St_heston_simu(1000)):.3f}")
    print(f"Monte-Carlo multi-path European Call:{eu_MC:.3f}")

    eu_BS = options.eu_call_BS(S0, 0)
    print(f"Black-Scholes European Call:{eu_BS:.3f}")
    
    f = lambda sigma: Options(K=K, sigma = sigma, r = r, T = T).eu_call_BS(S0, 0) - european_C
    brentq(f, 0.5, 1)






















