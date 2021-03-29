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
    delta[n:] = data['X'].values[n:] - data['X'].values[:-n]
    tag = '$ dX_{' + str(n) + '} $'
    data[tag] = delta
    return tag

def calc_deltas(data, deltas = [1]):
    tags = []
    for i in deltas:
        tags.append(calc_delta(data, i))
    return tags


path = 'C:/Users/user/Desktop/Нужная всячина/Projects/COVID/Data/COVID-19/'
fig, ax = plt.subplots()

country = 'Belarus'
left = 1000
right = 70000
values = [1, 3, 7, 10, 14, 21, 28]
# Data preparation.
data, country = read(country, with_recovered=True, path = path)
deltas = pd.DataFrame(data['X'])
deltas['I'] = data['X'] - data['R']
tags = calc_deltas(deltas, values)
deltas = deltas[(deltas['X'] > left) & (deltas['X'] < right)]

legend = ['I']
ax.plot(deltas['X'], deltas['I'])
for tag in tags:
    coef = deltas['I'].max() / deltas[tag].max()
    deltas[tag] = deltas[tag].values * coef
    print('Product coefficient for {0} = {1:.8f}'.format(tag, coef))

    ax.plot(deltas['X'], deltas[tag], linewidth=1)
    legend.append(tag)
ax.legend(legend)
plt.title(country)
plt.savefig('results/{0}_deltas.pdf'.format(country))
