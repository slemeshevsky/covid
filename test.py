# -*- coding: utf-8 -*-

from covid.utils import read

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd
import patsy as pt
import sklearn.linear_model as lm

def regr_coef(x, y):
    skm = lm.LinearRegression()
    skm.fit(x, y)
    beta0 = skm.intercept_
    beta1 = skm.coef_[0]
    return beta0, beta1

path = 'C:/Users/user/Desktop/Нужная всячина/Projects/COVID/Data/COVID-19/'
fig, ax = plt.subplots()
country = 'Belarus'
style = {"ls" : 'none',
         "marker" : 'o',
         "markersize" : '3'}
left = 10000
bound = 22000
right = 65000
# Getting data from csv.
data, country = read(country, with_recovered=True, path=path)
data['I'] = data['X'] - data['R']
data['alpha'] = data['dR']/data['I']
data['dXbyI'] = data['dX']/data['I']
# Data preapatarion.
regr_data = data.rolling(7).mean().dropna()
regr_data['I'] = regr_data['X'] - regr_data['R']
regr_data['alpha'] = regr_data['dR']/regr_data['I']
regr_data['dXbyI'] = regr_data['dX']/regr_data['I']
regr_data = regr_data[(regr_data['X'] > left) & (regr_data['X'] < right)]
data_left = regr_data[regr_data['X'] <= bound]
data_right = regr_data[regr_data['X'] > bound]
data = data[(data['X'] > left) & (data['X'] < right)]

# Beta
ax.plot(regr_data['X'], regr_data['dXbyI'])
# Linear regression.
    # <= bound
beta0_left, beta1_left = regr_coef(x = data_left['X'].values.reshape(-1, 1), 
                                  y = data_left['dXbyI'].values)
x = regr_data[regr_data['X'] < 33000]['X'].values
y = beta1_left * x + beta0_left
ax.plot(x, y, '-.')
beta = -beta1_left
N = beta0_left / beta
print('Linerar regression for data <={0}: beta = {1:.10f}; N = {2:.1f}'.format(bound, beta, N))
    #  > bound
beta0_right, beta1_right = regr_coef(x = data_right['X'].values.reshape(-1, 1), 
                                    y = data_right['dXbyI'].values)
x = regr_data['X'].values
y = beta1_right * x + beta0_right
ax.plot(x, y, '--')
beta = -beta1_right
N = beta0_right / beta
print('Linerar regression for data  >{0}: beta = {1:.10f}; N = {2:.1f}'.format(bound, beta, N))
# Finding intersection point.
x_int = (beta0_left - beta0_right) / (beta1_right - beta1_left)
y_int = beta0_left + beta1_left * x_int
y_int_test = beta0_right + beta1_right * x_int
print('Interseption point = ({0:.8f}, {1:.8f}), tesing y = {2:.8f}'.format(x_int, y_int, y_int_test))
#print(data[data['X'] > x_int].index)
print('Nearest datas:')
print(data[(data.index == '2020-05-07') | (data.index == '2020-05-06')])
# Plotting.
ax.set_ylim((0.01, 0.1))
ax.legend(['$\\frac{\dot{X}}{I}$ smoothed', 
           '$-\\beta X + \\beta N$ $ (\leq {0}) $'.format(bound), 
           '$-\\beta X + \\beta N$ $ (>{0}) $'.format(bound)])
plt.title(country)
plt.savefig('results/{2}_dXbyI_{0}_{1}.pdf'.format(left, right, country))

# Alpha
fig, ax = plt.subplots()
ax.plot(regr_data['X'], regr_data['alpha'])
# Linear regression.
beta0, beta1 = regr_coef(x = regr_data['X'].values.reshape(-1, 1), 
                    y = regr_data['alpha'].values)
x = regr_data['X'].values
y = beta1 * x + beta0
ax.plot(x, y, '--')
# Plotting.
ax.legend(['$\\frac{\dot{R}}{I}$ smoothed', 
           '$\\alpha = \\beta_0 + \\beta_1 x$'])
plt.title('{2}: $\\beta_0 = {0:.8f}$, $\\beta_1 = {1:.8f}$'.format(beta0, beta1, country))
plt.savefig('results/{2}_alpha_{0}_{1}.pdf'.format(left, right, country))
