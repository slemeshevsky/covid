# -*- coding: utf-8 -*-

from covid.utils import read

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd
import patsy as pt
import sklearn.linear_model as lm


def calc_delta(data, n = 1):
    delta = np.zeros_like(data['X'].values)
    #print(delta[n:].size)
    #print(data['X'].values[1:-n].size)
    #print(data['X'].values[:-n-1].size)
    if n == 1:
        delta[n:] = data['X'].values[n:] - data['X'].values[:-n] # \Delta X_k(t) = X(t) - X(t-k)
    else:
        delta[n:] = data['X'].values[1:-n+1] - data['X'].values[:-n] # \Delta X_k(t) = X(t-k+1) - X(t-k)
    tag = '$ dX_{' + str(n) + '} $'
    data[tag] = delta
    return tag

def calc_deltas(data, deltas = [1]):
    tags = []
    for i in deltas:
        tags.append(calc_delta(data, i))
    return tags


path = 'C:/Users/user/Desktop/Нужная всячина/Projects/COVID/Data/COVID-19/'

country = 'Belarus'
left = 1000
right = 70000
values = np.array([10, 15, 20, 25, 30])
# Data preparation.
data, country = read(country, with_recovered=True, path = path)
deltas = pd.DataFrame(data['X'])
deltas['I'] = data['X'] - data['R']
tags_deltas = calc_deltas(deltas, range(1, 30))
deltas = deltas[(deltas['X'] > left) & (deltas['X'] < right)]
print(deltas)

tags_regr = []
for val in values:
    tags_regr.append('$ I_{' + str(val) + '} $')
coefs = pd.DataFrame(np.zeros((values.max(), values.size)), index=range(1, values.max()+1), columns=tags_regr)

fig, ax = plt.subplots()
legend = ['I']
ax.plot(deltas['X'], deltas['I'])
# Plotting regression from X.
skm = lm.LinearRegression(fit_intercept=False, positive=True)
for n in values:
    tag = '$ I_{' + str(n) + '} $'
    x = pd.DataFrame(deltas, columns=['X', *reversed(tags_deltas[:n-1])]).to_numpy()
    y = deltas['I'].values
    skm.fit(x, y)
    result = np.zeros_like(y)
    for i in range(n):
        result += skm.coef_[i]*x[:,i]
    coefs[tag][:n] = skm.coef_
    deltas[tag] = result

    legend.append(tag)
    ax.plot(deltas['X'], deltas[tag], linewidth=1)
ax.legend(legend)
plt.title(country)
plt.savefig('results/{0}_regrIfromX.pdf'.format(country))
# Plotting regression from time.
fig, ax = plt.subplots()
legend = ['I']
ax.plot(deltas.index, deltas['I'])
for n in values:
    tag = '$ I_{' + str(n) + '} $'
    legend.append(tag)
    ax.plot(deltas.index, deltas[tag], linewidth=1)
ax.legend(legend)
plt.title(country)    
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.2)
plt.savefig('results/{0}_regrIfromT.pdf'.format(country))
# Plotting coefficients.
fig, ax = plt.subplots()
for tag in tags_regr:
    ax.plot(coefs.index, coefs[tag])
ax.legend(tags_regr)
plt.title('Coefficients of regression')
plt.savefig('results/{0}_regrCoeffs.pdf'.format(country))
