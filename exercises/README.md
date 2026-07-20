## Exercise

### Uniformly random i.i.d. particles: bouncing leads to indistinguishable passing-through
- 2 ants on a stick; 100 bugs on a stick: 
- $max(x_1, x_2, ..., x_n); $
- $E[M] = \int_{0}^{1} t*p(t)dt$ 
- where p(t) defines probability density function


### Choices available
- passengers with one lost ticket to seat uniformly random
    - first passenger seating situation + all rest passengers
    - seating decision prior to passenger k only delays a mis-seating decision and not suspends the process
    - passenger k's options 
- best candidate selection
    - given the best candidate is at k, the best will be selected if secondary-best candidate prior to k is within the rejection window
    - reindex summation bounds to align with Harmonic Series
- seating with no adjacent neighbors
    - first audience seating situation + all rest audience
    - $1 + \sum_{i=1}^{n} max(0,left seatings) + max(0,rightseatings) = 1 + \frac{2}{n}$ * $\sum_{i}^{n-2} f(i)$

### Payoff, card game and coin-flipping game
- stop game when reach repetitive value {1, 2, 3, 1, 2, 3}
    - expected payout given last-roll result
    - earnings at each round
- with Tail appears, keep current payout or forfeit it to play one more round
    - N: total number of flips til the first tail, **inclusive** of the tail flip; N~Geom($\frac{1}{2}$)
    - $E[Heads] = \frac{1}{p} - 1$
    - situation to forfeit: first-round payout < $E[h_{payout} * (N-1)]$
        - where $E[h_{payout} * (N-1)]$ measures a fresh-round payout since hasn't flipped a tail
        - as first-round turns to be Head, it's equivalent to start the game fresh
      
### Optimal stopping 
- stop and collect $x_t$ or reach $x_n$ and forced to take $x_n$
    - stopping cutoff, $max (x_t, V_{t+1})$
    - $E[max(x_t, V_{t+1})] = \int_{0}^{V_{t+1}} V_{t+1} + \int_{V_{t+1}}^{1} x dx$      ~ Bellman Equation
- with two fair dices, when both dices don't roll out 1, accumulate face value to running total; if either dice rolls a 1, the game stops and lose the sum.
    - running sum to be E[Sum + expected-2-dice-sum] = P(2 dices not rolling 1) * E[Sum + 2* $\frac{2+3+4+5+6}{5}$ ] = E[Sum + 2*4] + P(2 dices rolling 1) * 0
      
### distinct-item set
- each of 5 boxes has 1 coupon drawn uniformly random without replacement. Number of boxes to acquire for collecting at least each of 5 coupons

- 4 distinct cards in a set, 6 independent set; E[unique cards in total]

- with 2 fair dices, E[larger of 2 numbers]
