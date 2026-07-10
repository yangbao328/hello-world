#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Thu Jul 9

@author: tianyang

Girsanov Theorem
    dMt = MtAtdBt, M0 = 1
  if Wt = Bt - integral_{0}^(t) A(s)ds
then Wt is a standard brownian motion and dBt = Atdt + dWt in Q measurement

Key concepts
        Bt in Mt-premis
        Bt in P-probability measurement
        Bt in Q-probability measurement
  Bt_tilde in Q-probability measurement
"""



























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

