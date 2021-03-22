# -*- coding: utf-8 -*-

from covid.utils import read

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd
import patsy as pt
import sklearn.linear_model as lm
# import statsmodels.api as sm


country = 'Belarus'
# Getting data from csv.
data, country = read(country, with_recovered=True)
data['I'] = data['X'] - data['Recovered']  # - data['Deaths']
data['alpha'] = data['dR']/data['I']
data['dXbyI'] = data['Delta']/data['I']
# Data preapatarion.
left = 10000
right = 65000
data = data.rolling(7).mean().dropna()
regr_data = data[(data['X'] > left) & (data['X'] < right)]

# Beta
# Linear regression.
x = regr_data['X'].values.reshape(-1, 1)
y = regr_data['dXbyI'].values
skm = lm.LinearRegression()
skm.fit(x, y)
beta = -skm.coef_[0]
N = skm.intercept_/beta
# Plotting.
fig, ax = plt.subplots()
regr_data[['X', 'dXbyI']].plot(x='X', ax=ax)
x = regr_data['X'].values
y = beta * (N - x) # -beta*x + beta*N
ax.plot(x, y)
ax.legend(['$\\frac{\dot{X}}{I}$', '$-\\beta X + \\beta N$'])
plt.title('Беларусь: $\\beta = {0:.8f}$, $N = {1:.0f}$'.format(beta, N))
plt.savefig('results/Belarus_dXbyI_{0}_{1}.pdf'.format(left, right))

# Alpha (cut data)
# Linear regression.
x = regr_data['X'].values.reshape(-1, 1)
y = regr_data['alpha'].values
skm = lm.LinearRegression()
skm.fit(x, y)
beta0 = skm.intercept_
beta1 = skm.coef_[0]
# Plotting.
fig, ax = plt.subplots()
regr_data[['X', 'alpha']].plot(x='X', ax=ax)
x = regr_data['X'].values
y = beta1 * x + beta0
ax.plot(x, y)
ax.legend(['$\\frac{\dot{R}}{I}$', '$\\alpha = \\beta_0 + \\beta_1 x$'])
plt.title('Беларусь: $\\beta_0 = {0:.8f}$, $\\beta_1 = {1:.8f}$'.format(beta0, beta1))
plt.savefig('results/Belarus_alpha_{0}_{1}.pdf'.format(left, right))

# Alpha (all data)
# Linear regression.
x = data['X'].values.reshape(-1, 1)
y = data['alpha'].values
skm = lm.LinearRegression()
skm.fit(x, y)
beta0 = skm.intercept_
beta1 = skm.coef_[0]
# Plotting.
fig, ax = plt.subplots()
data[['X', 'alpha']].plot(x='X', ax=ax)
x = data['X'].values
y = beta1 * x + beta0
ax.plot(x, y)
ax.legend(['$\\frac{\dot{R}}{I}$', '$\\alpha = \\beta_0 + \\beta_1 x$'])
plt.title('Беларусь: $\\beta_0 = {0:.8f}$, $\\beta_1 = {1:.8f}$'.format(beta0, beta1))
plt.savefig('results/Belarus_alpha_full.pdf')


#fig, ax = plt.subplots()
#fig.suptitle(country)
#sns.regplot(x=np.arange(len(data)), y='alpha', data=data, ax=ax, lowess=True)
#ax.set_title('$\\frac{\\dot{R}}{I}$')
#ax.set_xlabel('Дни с начала эпидемии в стране')
#ax.set_ylabel('') # $\\frac{\\dot{R}}{I}$')
#ax.set_ylim((0, 0.2))
#plt.savefig('results/{}_alpha.pdf'.format(country))

#fig, ax = plt.subplots()
#sns.regplot(x='X', y='dXbyI', data=regr_data, ax=ax, lowess=True)
#ax.set_title('$\\frac{\\dot{X}}{I}$')
#ax.set_xlabel('$X$')
#ax.set_ylabel('') # $\\frac{\\dot{X}}{I}$'
#ax.set_ylim((0, 0.15))

#plt.savefig('results/{}_dX_I.pdf'.format(country))
