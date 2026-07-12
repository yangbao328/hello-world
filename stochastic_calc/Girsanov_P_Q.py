#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modified on Thu Jul 9

@author: tianyang
Given conversion, Girsanov theorem, of a with-drift brownian motion to 
shiftless BM in Q-measurement,
it enables to model process under one risk-free rate for discounting

single risk-free rate to represent the stock price's drift is essential 
because otherwise different participants would model their expectation differently 
hence the market doesnt have a universal price for one stock; 
then with single risk-free rate in Q-measurement, 
it aggregates market participants as "market" and 
hence the market price of risk theta


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

"""

import numpy as np

S0 = 100; K = 90; mu = 0.1; r = 0;  sigmaSt = 1.0; #sigma not to be deltaT; consider which entity this sigma has influence on
T = 1; sigmaBt = T**0.5; n_paths = 1000

#Stock price with mu = 0.1
#Stock price with free-constant sigmaSt
#St~(mu, sigmaSt)
#Underlying Bt follows N~(0, T) since Bt-terminal to calcualte St-terminal
Z = np.random.default_rng().normal(0, sigmaBt, n_paths)         # Monte Carlo
Bt = Z * T**0.5                                                 # N~(0,deltat)
St = S0 * np.exp((mu - 0.5 * sigmaSt**2) * T + sigmaSt * Bt)              

#Compute Mt = exp(theta*Bt- 0.5*theta*deltaT) where theta = (r-mu)/sigma = -(mu-r)/sigma
#Mt, likelihood ratio, dQ/dP""
theta = (r - mu)/sigmaSt
Mt = np.exp(theta*Bt - 0.5*theta**2*T)

#Weight the discounted payoff by Mt and average
payoff = np.maximum(St-K,0)
price = np.exp(-r*T) * np.mean(payoff * Mt)
print(price)

#NOTES
#Incorrect-small price is calculated because
#a. sigmaSt and sigmaBt was not distinguished and both mistake (T/N)**0.5
#b. mistaking 1.sigmaBt to be intermediate-dBt variance (T/N)**0.5 is flawed 
#               because 1. intermediate-dBt to simulate full path is implemented with deltat**0.5
#                          while terminal-dBt variance is T
#             2.sigmaSt is a free constant representing annualised stock volatility
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

To find the likelihood shifting factor, 
P(delta Bt for up) = P(delta Bt for down) = P((deltat)**0.5) = 1/2 under P
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

