#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:44:14 2026

@author: tianyang
# Monte Carlo Verification 11June2026
Generate samples from t-distribution derived from Normal distribution and 
   Chi-squared distribution relationship. Plot PDF against the analytical formula. 
   Verify Central Limit Theorem by plotting distribution means from lognormal distribution. 
   
   t-distribution: Z/((Chi-squared/df)^0.5)

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm

random_eng = np.random.default_rng()

#normals = random_eng.normal(27, 3, 100) #to be N~(0,1) why and what if not
#normal_single = random_eng.normal(0,1) #this generate single value per normal distribution

#chi_squared = np.sum(normals**2) what does this represent? sum of all 100 elements^2 from normal avove; a chi-squared at~(27,3)

df = 3 #having degree of freedom, each of 100 samples is 3-element long
sample = 10000
normal_samples = random_eng.standard_normal(size=(sample,1)) 
#(sample, df=1) here df represents number of column? why have to be 1
#because normal distribution is a fixed distribution, has df only as 1, "single normal"

normal_chi = random_eng.standard_normal(size=(sample, df))


chi_square_samples = np.sum(normal_chi**2, axis=1, keepdims=True)#.reshape(-1, 1)
#Normal and Chi-squared here are dependent
#these two are dependent because Chi-squared uses the same value from Normal
#t-distribution requires Normal and Chi-squared to be independent
#+axis=0 has summation over column, resulting [1x3] Chi-squared matrix as opposed to [100x1? or 100x3] -> axis=1

t_value = normal_samples/np.sqrt(chi_square_samples/df) 
#why didnt this different shape vector division prompt error? 
#Dimensions from Normal and Chi-squared satisfied that Chi-squared dimension 'from the right is less or equal to Normals
#Numpy applied broadcasting where elements in Normal per column is divided by the same Chi-squared value
#in fact, each element in Normal should be divided by distinct Chi-squared value

#why was Chi-Squared in df = 3 rather than df = 1?

#t_samples = normal_samples[:, 0] / np.sqrt(chi_square_samples / df)  # why only take first column as Z?
#Numerator, the standard normal N~(0,1) has no degree of freedom and is of one dimension

np.allclose(np.percentile(t_value,95),scipy.stats.t.ppf(0.95, df=3))
plt.hist(t_value.flatten(),bins=100,density=True, alpha=0.6)
x = np.linspace(-3, 3,200)
plt.plot(x,scipy.stats.t.pdf(x,df))
#plt.hist(t_samples.flatten(), bins=100, density=True, alpha=0.6, label='Simulated t-samples')


"""Monte Carlo Cont'd 
Verify Central Limit Theorem by plotting distribution means from lognormal distribution. 
"""
# Lognormal, Y = e^(mu+sigma*Z), x>0

mu = 0
sigma = 1
sample = 1000 #number of lognormal sample to be run
sample_to_mean = [5,20,100] #number of observations included in each lognormal simulation, then use to calculate a mean  of that run


def lognormal_func(n_sample, n_observation, method):
    if method == 1:
    #Lognormal PDF, (relative) likelihood/density of value meeting x following lognormal distribution with associated sigma and mu 
    #MUST Integrate over dx to get the probability: f(x)*deltaX
    #NOT A SAMPLING METHOD
        x = np.linspace(0.1, 3, n_sample)
        lognormal = 1/(np.sqrt(2*np.pi) * x * sigma)*np.exp(-(np.log(x)-mu)**2/(2*sigma**2))
        return x,lognormal
    
    if method == 2:
    #Lognormal from Normal Random Samples
        normal_samples = random_eng.standard_normal(size=(n_sample, n_observation)) 
        estimated_mean = np.exp(mu + sigma * normal_samples).mean()
        return estimated_mean

    if method == 3:
        lognormal_samples = random_eng.lognormal(mean=mu, sigma=sigma, size=(n_sample, n_observation))
        return lognormal_samples



#lognormal = (1/(2*np.sqrt(np.pi))) * normal_samples * (-np.power(np.log(normal_samples),2)/2)
#INCORRECT: a. normal_samples contains negative value that will go into np.log, which produced nan term; 
#              support of lognormal distribution is x > 0
#           b. incorrect placement for "-" and formula expression
true_mean = np.exp(mu + sigma**2/2)
true_std = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2))

fig, axs = plt.subplots(1, 3, figsize=(7,4))
for i, n in enumerate(sample_to_mean):
    data = lognormal_func(sample, n, method=3)
    means = data.mean(axis=1)
    axs[i].hist(means, bins = 50, density = True, alpha = 0.6)
    axs[i].axvline(lognormal_func(sample, n, method=2), color='black',linestyle='--', label='sample estimate')
    axs[i].axvline(true_mean, color='coral',linestyle='--')
    #axs[i].plot(x,lognormal_func(sample, n, method=1)) 
    #overlaying lognormal is misunderstood concept
    #because CLT describes with larger n, the distribution of means follows Normal Distribution
    
    se = true_std / np.sqrt(n) #standard error
    x_norm = np.linspace(true_mean - 4*se, true_mean + 4*se, 200)
    axs[i].plot(x_norm, norm.pdf(x_norm, true_mean, se), 'r-', lw=2, label='Normal approx.')
    
    axs[i].set_title(f'Sample size = {n}')
    
plt.suptitle('CLT: Lognormal -> Normal')
plt.tight_layout()
plt.show()

#VERIFICATION of Lognormal Distribution and PDF
simulated_density = lognormal_func(10000, 1, method=3)  # 10000×1 array, simulation
plt.hist(simulated_density.flatten(), bins=80, density=True, alpha=0.6)
x, calc_PDF = lognormal_func(300, 1, method=1)
plt.plot(x, calc_PDF)  # PDF curve
plt.title('Lognormal using PDF formula versus simulation using np.exp()')
plt.xlim(0, 5)
plt.show()



