# -*- coding: utf-8 -*-

from covid.utils import read

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd
import patsy as pt
import sklearn.linear_model as lm

path = 'C:/Users/user/Desktop/Нужная всячина/Projects/COVID/Data/COVID-19/'
country = 'Belarus'
style = {"ls" : 'none',
         "marker" : 'o',
         "markersize" : '3'}
data, country = read(country, with_recovered=True, path=path)
# Data preparation.
data = data[data['X'] > 10000]
data['I'] = data['X'] - data['R']
regr_data = smooth_average(data, 7)
regr_data['I'] = regr_data['X'] - regr_data['R']
# Plotting.
vars = ['dX', 'dR']
for var in vars:
    fig, ax = plt.subplots()
    tag = var + 'byI'
    tag_tex = '$\\frac{\dot{' + var[1] + '}}{I}$'
    data[tag] = data[var]/data['I']
    regr_data[tag] = regr_data[var]/regr_data['I']
    ax.plot(data['X'], data[tag], **style)
    ax.plot(regr_data['X'], regr_data[tag])
    ax.legend([tag_tex, tag_tex + ' smoothed'])
    plt.title(country)
    plt.savefig('results/{0}_{1}_smoothed.pdf'.format(country, tag))
