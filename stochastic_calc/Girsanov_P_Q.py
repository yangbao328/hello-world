#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Thu Jul 9

@author: tianyang

Girsanov Theorem
    dMt = MtAtdBt, M0 = 1
  if Wt = Bt - integral_{0}^(t) A(s)ds
then Wt is a standard brownian motion and dBt = Atdt + dWt in Q measurement.
  log(Mt) = 
  Product_{i=1}^{n}  (1+-A(t)*deltat**0.5)
                  =  A(t)Wt - A(t)**2 Sum_{i=1}^{n} (deltat/2)
                  =  A(t)Wt - A(t)**2 * T/2
  Mt = exp(A(t)Wt - A(t)**2 * T/2)
dSt = u*St*dt + sigma*St*dWt^P 
dSt = r*St*dt + sigma*St*dWt^Q
dWt^P = dWt^Q + theta dt

dSt = u*St*dt + sigma*St*(dWt^Q + theta*dt)
dSt = u*St*dt + sigma*St*theta*dt + sigma*St*dWt^Q 
    = (u+sigma*theta)*St*dt + sigma*St*dWt^Q
u+sigma*theta = r => theta = (r-mu)/sigma

Key distinctions
        Bt in Mt-premise
        Bt in P-probability measurement
        Bt in Q-probability measurement
  Bt_tilde in Q-probability measurement
  
  
Simulate a stock under the physical-probability measure P with mu = 0.1
Implement the likelihood-ratio weighting to price a call under Q

"""

import numpy as np


S0 = 100; K = 90; mu = 0.1; r = 0; 
T = 1; sigma = 1.0; n_paths = 1000


#Stock price with mu = 0.1
Z = np.random.default_rng().normal(0, 1, n_paths)               # Monte Carlo
Bt = Z * T**0.5                                                 # N~(0,deltat)
St = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * Bt)              

#Compute Mt = exp(theta*Bt- 0.5*theta*deltaT) where theta = (r-mu)/sigma = -(mu-r)/sigma
#Mt, likelihood ratio, dQ/dP""
theta = (r - mu)/sigma
Mt = np.exp(theta*Bt - 0.5*theta**2*T)

#Weight the discounted payoff by Mt and average
payoff = np.maximum(St-K,0)
price = np.exp(-r*T) * np.mean(payoff * Mt)
print(price)


#%%


"""
Created on Tue Jul  7 21:06:46 2026

ORGANIZE IN MARKDOWN FILEs

Girsanov Theorem
With standard Brownian Motion Bt~N(0, deltat) that is under physical probability P, 
to have a brownian motion that has drift A(t), Bt can be shifted to follow N~(A(t), deltat)
to reach Bt_tilde by introducing likelihood of probabilities under P to it of under Q;
multiplication of detla Bt_tilde at all timestamps is a martingale under P, 
where dMt = AtMtdBt.

To find the likelihood shifting factor, P(delta Bt for up) = P(delta Bt for down) = P((deltat)**0.5) = 1/2 under P
supposing q is the probability for delta Bt_tilde upward, 1-q is it for delta Bt_tilde downward
q*deltat**0.5 + (1-q)*(-deltat**0.5) = A(t)deltat
q=0.5*(1+A(t)*deltat**0.5)
then likelihood at single step of Prob under Q to Prob under P is q/(1/2) = 1+A(t)*deltat**0.5.

To find total likelihood of Prob under Q to under P, 
Product_{i=1}^{n}  (1+-A(t)*deltat**0.5)

LN(Product_{i=1}^{n}  (1+-A(t)*deltat**0.5))
=Sum_{i=1}^{n} (log(1+-A(t)*deltat**0.5))
Taylor Expansion log(1+x) = x - x**2/2 + O(x**3)
=Sum_{i=1}^{n} (+-A(t)*deltat**0.5 - (A(t)*deltat**0.5)**2/2) where A(t) is a factor and deltat**0.5=dWt
=Sum_{i=1}^{n} (+-A(t)*deltat**0.5) - Sum_{i=1}^{n} ((A(t)*deltat**0.5)**2/2) 
=Sum_{i=1}^{n} A_i-1 * deltaBti
=A(t)Wt - A(t)**2 Sum_{i=1}^{n} (deltat/2)
=A(t)Wt - A(t)**2 * T/2

Product_{i=1}^{n}  (1+-A(t)*deltat**0.5)
=EXP(LN(Product_{i=1}^{n}  (1+-A(t)*deltat**0.5)))
=EXP(Wt - A(t)**2*T/2)
=Mt

CHECK why is total likelihood the martingale 

ERRONEOUS q*deltat + (1-q)*(-deltat) = A(t)deltat
ERRONEOUS q*A(t)deltat**0.5 + (1-q)*(-A(t)deltat**0.5) = A(t)deltat
ERRONEOUS (q*A(t)(deltat)**0.5)+(1-q)*(A(t)(deltat)**0.5) = A(t)*deltat**0.5
CHECK where A(t)delta t is the jump size with drift and separately expectation of single jump size with drift under Q

"""

