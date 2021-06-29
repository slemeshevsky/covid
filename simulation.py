# -*- coding: utf-8 -*-

from covid.CountryDataProcessor import CountryDataProcessor as CDP
from covid.cdp_fabric import get_country_data_processor
from covid.plotting import plot_weights, plot_infected
from covid.preprocess import smooth_average
from covid.utils import read, calc_deltas

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy as pt
import seaborn as sns
import sklearn.linear_model as lm

# Ломанная до 10, от 10 до 16. На 16 двигать точку вверх-вниз

def lin_coefs(data, cdp, N=40, country='Belarus'):
    deltas, features, tex_features = cdp.calc_deltas(N, data.index)

    for l in range(12, 17):
        coef_ = np.ones(len(features))
        h = 1/(N+1-l)
        coef_[l:] = np.array([i*h for i in range(1, N+1-l)][::-1])
        y_lin = deltas[features].dot(coef_)

        data['Lin_regr'] = y_lin
        fig, ax = plt.subplots()
        legend = ['I', 'Lin']
        data[['X', 'I', 'Lin_regr']].plot(x='X', ax=ax)
        ax.legend(legend)
        plt.title('{0}, $l={1}$'.format(country, l))
        plt.savefig('results/{0}_simulation_lin_X_l{1}.png'.format(country, l))

def p_lin_coefs(data, cdp, h=None, l1=10, l2=17, N=40, country='Belarus'):
    deltas, features, tex_features = cdp.calc_deltas(N, data.index)
    coef_ = np.ones(len(features))
    h = h if h is not None else 0.5
    dh1 = (1-h)/(l2-l1)
    dh2 = h/(N-l2)
    coef_[l1:l2+1] = np.array([1-i*dh1 for i in range(0, l2-l1+1)])
    coef_[l2:] = np.array([h-i*dh2 for i in range(1, N-l2+1)])
    y = deltas[features].dot(coef_)
    data['P_Lin_regr'] = y
    legend = ['I', 'P_Lin']

    fig, ax = plt.subplots()
    data[['X', 'I', 'P_Lin_regr']].plot(x='X', ax=ax)
    ax.legend(legend)
    plt.title('{0}, $h={1}$'.format(country, h))
    plt.savefig('results/{0}_simulation_p_lin_X_h{1}.png'.format(country, h))
    return coef_


def exp_coefs(data, cdp, N=40, country='Belarus', q=0.9, tol=1e-5):
    deltas, features, tex_features = cdp.calc_deltas(N, data.index)
    coef_ = {}
    for l in range(5, 17):
        exp_coef_  = [1.0 for _ in range(l)]
        right_bound = l
        b = 1.
        while(b > tol):
            right_bound += 1
            b *= q
            exp_coef_.append(b)

        coef_[l] = exp_coef_
        deltas, features, tex_features = cdp.calc_deltas(right_bound, data.index)
        y_exp = deltas[features].dot(np.array(exp_coef_))
        data['Exp_regr'] = y_exp
        fig, ax = plt.subplots()
        legend = ['I', 'Exp']
        data[['X', 'I', 'Exp_regr']].plot(x='X', ax=ax)
        ax.legend(legend)
        plt.title('{0}, $l={1}$, $q={2}$'.format(country, l, q))
        plt.savefig('results/{0}_simulation_exps_X_l{1}_q_{2}.png'.format(country, l, q))
    return coef_

def main(q=0.9):
    country = 'Belarus'
    num_wave = 1
    N = 40
    cdp, country_info = get_country_data_processor(country)
    data_cut = cdp.get_wave(num_wave)
    return exp_coefs(data_cut, cdp, q=q)
    # lin_coefs(data_cut, cdp)


if __name__ == '__main__':
    main()

# idx = [q*(l-i) for i in range(l, N)]
# exp_coef_[l:] = np.exp(idx)
# y_exp.plot(ax=ax)
