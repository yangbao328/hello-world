#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 20:49:53 2026

@author: tianyang
"""

import numpy as np
import matplotlib.pyplot as plt
from the_engine import Market, Options, Hedger
    

#P&L impact under transaction cost 
'''
Transaction cost is applied at initial state, rebalancing and final state
'''

    

def PnL_rebalance(N_rebalance, N_simulation):
        
    pnl_list = []
    S0 = 100; K = 90; r = 0; sigma = 1; T = 1; N = N_rebalance; transaction_cost = .001

    #In 252 business days, simulate hedger P&L for 1000 times and verify their mean 
    for _ in range(N_simulation):
        market = Market(S0 = S0, r = r, sigma = sigma, T = T, N = N)
        options = Options(K=K, T=T, r=r, sigma=sigma)
        hedger = Hedger(market, options, trz_cost=transaction_cost)
        hedger.rebalance()
        pnl = hedger.final_trade()
    
        pnl_list.append(pnl)
        
    print(f'Mean P&L is {np.mean(pnl_list):.3f}')
    print(f'Standard deviation P&L is {np.std(pnl_list):.3f}')
    
    #plt.scatter(range(1000), pnl_list, s=.7)
    plt.hist(pnl_list, bins=50, density=True, alpha=0.7)
    plt.axvline(np.mean(pnl_list), color='pink', linestyle='--', label=f'Mean = {np.mean(pnl_list):.3f}')
    plt.xlabel('Simulated PnL')
    plt.ylabel('Probability Density of PnL')
    plt.title('P&L Distribution')
    plt.legend()
    plt.show()

PnL_rebalance(252, 1000)