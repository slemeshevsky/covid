# -*- coding: utf-8 -*-

from covid.utils import read

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd
import patsy as pt
import sklearn.linear_model as lm
import statsmodels.api as sm


data, country = read('Belarus', with_recovered=True)
data['I'] = data['X'] - data['Recovered']  # - data['Deaths']
data['alpha'] = data['dR']/data['I']
data['dXbyI'] = data['Delta']/data['I']

regr_data = data[(data['X'] < 52000) & (data['dXbyI'].notnull())]
x = regr_data['X'].values.reshape(-1, 1)
y = regr_data['dXbyI'].values
skm = lm.LinearRegression()
skm.fit(x, y)
print(skm.intercept_, skm.coef_)


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
