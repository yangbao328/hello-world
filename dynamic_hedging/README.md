# Dynamic Hedging Engine

This dynamic delta-hedging engine is consisted of stock simulation, European call option pricer and delta, and hedging operations through time. The hedger is in role of counterparty of European-call purchaser, therefore holds short-call and long-security positions to offset fluctuations from the call purchase.  

This project verifies with r=0, the average PnL is around 0, and explores impacts of transaction cost and hedging threshold has on simulated PnL.

# Motivations

Stock simulation follows geometric brownian motion and is vectorized via log-normal transformation for computation efficiency; European call pricer is implemented following Black-Scholes formula under risk-neutral measurement Q, of which risk-free rate is applied for discount and interest, and an arbitrage-free replication is allowed to be feasible. In realized executions, this project simulates frictions such as transaction cost and illustrates trade-off between reducing cost and un-hedged exposure. 

Demonstrations this project provides are
<br>a. zero-mean of simulated PnL under risk-neutral measurement;
<br>b. negative impact of proportional transaction cost;
<br>c. trade-off between cost reduction and un-hedged delta exposure via hedging threshold.

# Mathematical Description

Stock price follows geometric Brownian Motion $dS_t = r S_t dt + \sigma S_t dB_t$ and is implemented with vectorised log-transformation Y = lnS_t. Option pricer assumes Black-Scholes formula, of which $C_t = e^{-rt} (FN(d_1) - KN(d_2))$.
<br><br>With short-call and long-security positions, initial position is funded by cash from shorting a call, and funds for rebalancing is borrowed at rate r while interests on cash is earned at rate r as well. Number of shares to be held at intermediate timestamp is measured of the option's delta, $N(d_1)$, and security-position rebalancing occurs at each step when hedging threshold is being met. Final PnL is computed in cash settlement, with security sold, interest accrued and call option pay-off rendered to purchaser reaching time T.
 
# Project Components

- **the_engine.py** contains three major classes
   - Market: Simulates stock price via Geometric Brownian Motion in vectorisation
   - Options: Computes call option price and delta at intermediate timestamps following Black-Scholes formula
   - Hedger: Reflects stock positions as delta fluctuates and according cash value at timestamps
- **hedging_PnL.py** contains simulation, visualize observations on transaction cost and hedging threshold

# Observation Summary

a. with zero-transaction cost and daily rebalancing frequency, distribution of simulated PnL follows bell-curve shape; its mean centers around 0 while standard deviation is around 2.
<div align="center">
<img src="assets/PnL_Distribution.png" alt="PnL Distribution_noTC, 1000 simulations" width="300" />
</div>
b. With transaction cost applied at initial state, rebalancing and final state, simulated PnL mean becomes negative (e.g., –1.52 for 0.1% cost) while the standard deviation remains similar.
<div align="center">
<img src="assets/PnL_Distribution.png" alt="PnL Distribution_TC, 1000 simulations" width="300" />
</div>
c. For a delta adjustment smaller than the threshold, the hedging trade is not executed
In experiment, the threholds looked at are [0, 0.005, 0.01, 0.02, 0.05]; from this rough estimation, the higher the threshold, the smaller magnitude of negative PnL yet the more volatile the PnL simulations appear to be.
With introduction of hedging threshold, the hedging is not as closely aligned, so it introduces more uncertainty into the option-stock replication due to unhedged exposure. As the threshold increases, the mean P&L improves (less negative) but the dispersion grows, reflecting greater un‑hedged risk,

<div align="center">
<img src="assets/Hedging_Thresholds.png" alt="Hedging Thresholds" width="300" />
</div>

# Limitations and Next Iterations

The stock is currently simulated at the same frequency as the hedge rebalancing. Separating the simulation time grid from the trading grid would allow more realistic modelling.
<br>a. separation of St simulation and rebalance frequency
<br>b. barrier option pricer
<br>c. vectorize hedger with bulk-St simulation
