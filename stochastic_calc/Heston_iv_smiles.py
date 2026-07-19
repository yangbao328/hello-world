#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:14:25 2026

@author: tianyang

Observations

Parameter: rho = 0.0, correlation of security-diffusion and security-volatility diffusion (random shock)
Initial attempt on plotting implied volatility was from fresh instances of Monte-Carlo Call pricer 
which is underpinned by individual referencing of St;for each K, St respectively is calcualted 
with uncorrelated underlying dWtS and dWtv, therefore the initial attempt introduced uncorrelated 
errors which generates seemingly random implied volatility at various Ks. 

Refined approach , with rho = 0.0, calculated one universal St as parameter at various K; resulted implied
volatility then displays the convex shape over strike Ks while of the same expiration date. 

with rho = -0.7             --- Leverage Effect, negative correlation
the negative parameter indicates a rise in volatility comes along when stock price falls.
On the left tail, with low-strike and likely low-St situation,
for an ITM call, the call price is valuable even though expected payoff of the call is reduced;
the call-price value is of higher-volatlity associated with lower-St, captured by rho in heston, 
which increases option's time value through gamma/convexity and therefore call price. 
Moreover, with low-St and high-volatility and fixed S0 and K, 
via put-call parity, C-P=S0-e^(-rT)K, 
the corresponding deep-OTM put increases call price as well; 
this higher-than-BS call price has the effect of generating higher implied volatility. 

On the left tail, since call-price difference between heston model and BS model is positively large,
implied volatility would take on larger value to catch higher heston-model price.
On the right tail, regarding the same ITM call, there're less St to reach such magnitude and
if St reaches, corresponding underlying volatility will be small to maintain the price, 
which decreases ITM call's time value; overall the heston-model price is lower and therefore 
requires lower volatility from BS to match the price.
These occassions with low strike on left tail and high strike on right tail produces
downward tilt implied volatility skew. 

"""
import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dynamic_hedging.the_engine import Options
from Heston_iv_montecarlo import Heston_Volatility

#from the_engine import Options


T = 1; N = 252; rho = -0.0; kappa = 0.1; theta = 0.12
sigmav = 0.5; v0 = .3; r = 0.1; S0=100; K=90; n_paths = 30000
sigma_low = 0.05; sigma_high = 1
strikes = np.arange(80, 135, 2)

heston = Heston_Volatility(T=T, N=N, rho=rho, kappa=kappa, theta=theta, 
                               sigma_v = sigmav, v0 = v0, r = r, S0 = S0)

ivols = []
calls = []
for K in strikes:
    
    ivols.append(heston.implied_vol(K, sigma_low, sigma_high, n_paths))
    calls.append(heston.European_Call_MC_simu(K, n_paths))

plt.plot(strikes, ivols, 'o-', markersize=3)
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.show()

#Volatility Skew and downward movement due to negative correlation rho
fig, ax1 = plt.subplots()
ax1.set_xlabel('Strike K')
ax1.set_ylabel('Implied Volatility', color='lightpink')
ax1.plot(strikes, ivols, color='lightpink', marker='o', markersize=2)
ax1.tick_params(axis='y', labelcolor='lightpink')

ax2 = ax1.twinx()
ax2.set_ylabel('Call, Heston', color='gray')
ax2.plot(strikes, calls, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

fig.tight_layout()
plt.title('Implied Volatility, individual MC-simulation at strike Ks')
plt.show()

#IV Smile and MC-simulation error reduced
St, iv_smiles = heston.implied_vol_smile(strikes, sigma_low, sigma_high, n_paths)


fig, ax1 = plt.subplots()
ax1.set_xlabel('Strike K')
ax1.set_ylabel('Implied Volatility', color='lightpink')
ax1.plot(strikes, iv_smiles, color='lightpink', marker='o', markersize=2)
ax1.tick_params(axis='y', labelcolor='lightpink')

ax2 = ax1.twinx()
ax2.set_ylabel('Call, Heston', color='lightseagreen')
ax2.plot(strikes, calls, color='lightseagreen')
ax2.tick_params(axis='y', labelcolor='lightseagreen')

fig.tight_layout()
plt.title('Implied Volatility, MC-error reduced')
plt.show()

#print(f"Risk-neutral expectation for St is {S0 * np.exp(r*T):.3f}")


# Verify call price and implied volatility 
for K in [70, 100, 130]:
    hp = heston.European_Call_MC_simu(K, n_paths)
    bp = Options(K=K, sigma = v0**0.5, r = r, T = T).eu_call_BS(S0, T-T)
    print(f"Strike at {K}, heston-call price {hp}, BS-call price {bp}")

#test on github username














