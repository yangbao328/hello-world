#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 19:06:27 2026

@author: tianyang

This dynamic hedging engine simulates 
stock movement, European-style options and Greeks evolvement, order execution and P&L monitoring
enabled by Market, Options and Hedger, the three fundamental classes. 

Goals
simulate cost of continuously delta‑hedging a short option equals the Black‑Scholes price,
Convey final P&L converges to zero (minus transaction costs) when the stock follows the risk‑neutral process.

Limitations
St simulation step shares N from rebalancing frequency
"""

import numpy as np
from scipy.stats import norm

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
        self.rng = np.random.default_rng(seed)                 #mean, sigma, size
        self.St_path = None

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
    
    def St_simulate(self):
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
        Y[1:] = np.log(self.S0) + np.cumsum(drift_Y+diffn_Y)# additive BM observed from dY that relies on initial dt and dBt
        
        self.St_path = np.exp(Y)
        
        #return self.St_path[step]
    
    def GBM_St(self, step):
        
        if self.St_path is None:
            self.St_simulate()
            return self.St_path[step]
            
        else:
            return self.St_path[step]
            
    def GBM_St_terminal(self):
        
        Z = self.rng.normal(0, 1)
        St = self.S0 * np.exp((self.mu - 0.5 * self.sigma**2) * self.T
                             + self.sigma * np.sqrt(self.T) * Z)
        return St

class Options:
    '''
    Goal
    Simulate current price of the option
    Illustrate current delta (and other Greeks)

    Notes
    Following risk-neutral measurement Q, Black-Scholes assumes drift-term of St is upon r
    Stock simulation in class Market shares risk-free rate r for St drift
    Ct = e^(-rt) (FN(d1)-KN(d2))
    '''
    
    def __init__(self, K, T, r, sigma, seed=None):
        
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
    
    def d1(self, St, t):
        
        nom = np.log(St/self.K) + (self.r+self.sigma**2/2)*(self.T-t)
    
        dem = self.sigma*(self.T-t)**0.5
        
        return nom/dem
        
    def eu_call_BS(self, St, t):
        
        d1 = self.d1(St,t)
        d2 = d1 - self.sigma*(self.T-t)**0.5
        
        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)
        
        #C = np.exp(-self.r*(self.T-t)) * (St*Nd1 - self.K*Nd2)
        C = St * Nd1 - self.K * np.exp(-self.r * (self.T - t)) * Nd2
        
        return C
    
    def Delta(self, St, t):
        
        d1 = self.d1(St, t)
        
        return norm.cdf(d1)

    
class Hedger:
    '''
    this hedger takes positions of shorting call and longing underlying secuirty
    
    Goal
        Reflect number of shares to purchase at timestamps as Delta fluctuates
        Collect interest on cash or log interest on loan
        Compute P&L when t reaches T
    '''
    def __init__(self, market, options):
        self.market = market                                #stock price at timestamps
        self.options = options                              #option price and delta for hedging purpose
        self.T = market.T
        self.dt = market.T/market.N
        self.cash = 0.0                                     #to simulate P&L
        self.stock_shares = 0.0                             #number of shares to hedge
        self.short_call = True                              #short call and long security
        #self.N_steps = N_steps                             #duplicated from options N
        
        #initiate T=0 as beginning state
        S0 = self.market.GBM_St(0)        
        target_delta = self.options.Delta(S0,0)
        self.cash = self.options.eu_call_BS(S0, 0)          #short call
        self.cash -= S0*target_delta                        #long stock
        self.stock_shares = target_delta

    def rebalance_step(self, step):
        

        self.cash *= np.exp(self.options.r*self.dt)
        
        St = self.market.GBM_St(step)
        target_delta = self.options.Delta(St, self.dt*step)
        hedged_delta= target_delta - self.stock_shares
        
        self.cash -= hedged_delta * St
        self.stock_shares = target_delta
        
    
    def rebalance(self):
        for step in range(1, self.market.N):
            self.rebalance_step(step)
        
    def rebalance_loop(self):  
        
        '''
        Rebalance and update for intermediate timestamps
        '''
        #book = {0:(self.stock_shares, self.market.GBM_St(0), self.cash)}
        
        for step in range(1,self.market.N):
            
            self.cash *= np.exp(self.options.r*self.dt)      #interest on loan from 0 to t1

            St = self.market.GBM_St(step)
            t = step*self.dt
            
            target_delta = self.options.Delta(St, t)
            
            hedged_delta = target_delta - self.stock_shares
            self.cash -= St * hedged_delta
            self.stock_shares = target_delta

    
    def final_trade(self):
        
        '''
        Position: short a call and long underlying security
        
        Closing:  accrue cash interest;
                  provide #security longed to the call pruchaser; 
                  call payoff included as part of cash position
        '''
        N = self.market.N
        St = self.market.GBM_St(N)
        
        self.cash *= np.exp(self.options.r*self.dt)         #accrue interest at T
        self.cash += St*self.stock_shares                   #close longing stock positions by selling at St
        #selling security and receive St

        if self.short_call == True:
            payoff = -max(St-self.options.K,0)                   
        #pay option purchaser short-call payoff (St-K)
        #!!!   np.max and np.maximum 
        #where np.max treats 0 as axis argument rather than floor
        
        self.cash += payoff
        #St-(St-K) = K, final P&L after cash settlement

        return self.cash
    
    
S0 = 100; K = 90; r = 0; sigma = 1; T = 1; N = 252

pnl_list = []

for _ in range(1000):
    market = Market(S0 = S0, r = r, sigma = sigma, T = T, N = N)
    options = Options(K=K, T=T, r=r, sigma=sigma)
    hedger = Hedger(market, options)
    hedger.rebalance()
    pnl = hedger.final_trade()

    pnl_list.append(pnl)
    
print(f'Mean P&L is {np.mean(pnl_list)}')
print(f'Standard deviation P&L is {np.std(pnl_list)}')
