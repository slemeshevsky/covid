# -*- coding: utf-8 -*-

import numpy as np
from covid.utils import read
import seaborn as sns
import matplotlib.pyplot as plt


data, country = read('Belgium', with_recovered=True)
data['I'] = data['X'] - data['Recovered']  # - data['Deaths']
data['alpha'] = data['dR']/data['I']
data['dXbyI'] = data['Delta']/data['I']


fig, ax = plt.subplots()
fig.suptitle(country)
sns.regplot(x=np.arange(len(data)), y='alpha', data=data, ax=ax, lowess=True)
ax.set_title('$\\frac{\\dot{R}}{I}$')
ax.set_xlabel('Дни с начала эпидемии в стране')
ax.set_ylabel('') # $\\frac{\\dot{R}}{I}$')
plt.savefig('results/{}_alpha.pdf'.format(country))

fig, ax = plt.subplots()
sns.regplot(x='X', y='dXbyI', data=data, ax=ax, lowess=True)
ax.set_title('$\\frac{\\dot{X}}{I}$')
ax.set_xlabel('$X$')
ax.set_ylabel('') # $\\frac{\\dot{X}}{I}$'

plt.savefig('results/{}_dX_I.pdf'.format(country))
