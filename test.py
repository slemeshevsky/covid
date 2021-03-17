# -*- coding: utf-8 -*-

from covid.utils import read

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd
import patsy as pt
import sklearn.linear_model as lm
# import statsmodels.api as sm


data, country = read('Belarus', with_recovered=True)
data['I'] = data['X'] - data['Recovered']  # - data['Deaths']
data['alpha'] = data['dR']/data['I']
data['dXbyI'] = data['Delta']/data['I']


left = 10000
right = 65000
regr_data = data[(data['X'] > left) & (data['X'] < right)]
x = regr_data['X'].values.reshape(-1, 1)
y = regr_data['dXbyI'].values
skm = lm.LinearRegression()
skm.fit(x, y)
beta = -skm.coef_[0]
N = skm.intercept_/beta

fig, ax = plt.subplots()
regr_data[['X', 'dXbyI']].plot(x='X', ax=ax)
x = regr_data['X'].values
y = -beta*x + beta*N
ax.plot(x, y)
ax.legend(['$\\frac{\dot{X}}{I}$', '$-\\beta X + \\beta N$'])
plt.title('Беларусь: $\\beta = {0:.8f}$, $N = {1:.0f}$'.format(beta, N))
plt.savefig('results/Belarus_{0}_{1}.pdf'.format(left, right))

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
