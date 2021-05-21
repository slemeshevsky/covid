# -*- coding: utf-8 -*-

from IPython.core.pylabtools import figsize
from covid.BelarusDataProcessor import BelarusDataProcessor
from covid.plotting import plot_weights, plot_infected
from covid.preprocess import smooth_average
from covid.utils import read, calc_deltas

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy as pt
import seaborn as sns
import sklearn.linear_model as lm

#plt.ioff()

def calc_deltas(data, wnds=[1]):
    tags = []
    tex_tags = []
    for i in range(wnds[-1]):
        tag = 'dX' + str(i+1)
        tags.append(tag)
        tex_tag = '$\\delta_{{{}}}$'.format(i+1)
        tex_tags.append(tex_tag)
        data[tag] = data['dX'].shift(periods=i)
        data[tag].fillna(0., inplace=True)
    return tags, tex_tags

def plot_weights(features, weights, norms, show_fig=False, sorting=True):
    fig, axs = plt.subplots(ncols=2)
    if sorting:
        sorted_weights = sorted(zip(weights, features, norms), reverse=True)
        weights = [x[0] for x in sorted_weights]
        features = [x[1] for x in sorted_weights]
        norms = [x[2] for x in sorted_weights]

    weights = weights
    features = features
    norms = norms

    sns.barplot(y=features, x=weights, ax=axs[0])
    axs[0].set_xlabel("Веса")
    axs[0].set_title("Коэффициенты регрессии")

    sns.scatterplot(x=norms, y=features, ax=axs[1])
    axs[1].set_title('$|| I - \delta_k$ ||')
    fig.suptitle('Окно = {}'.format(len(features)))
    plt.savefig('results/Belarus_regrCoeffs_wind_{}'.format(len(features)))
    if show_fig:
        plt.show()
    plt.close()

def plot_infected(model, x_train, y_train, n):
    y_pred = model.predict(x_train)
    coef_first = model.coef_.copy()
    coef_second = model.coef_.copy()
    coef_first[n:] = 0.0
    coef_second[:n] = 0.0
    y_pred_1 = np.dot(coef_first, x_train.T)
    y_pred_2 = np.dot(coef_second, x_train.T)

path = '../COVID-19/'

def bel_plot_IdRDx():
    # Построение совместного графика I, I_theor, \Delta R и \Delta X, наиболее приближённого к I.
    num = 42
    alpha = 0.025

    bdp = BelarusDataProcessor()
    bdp.read_data(with_smooth=True)

    data_cut = bdp.get_wave()
    delta, features, tex_features = bdp.calc_deltas(num, data_cut.index)

    X_train = delta[features[::-1]].copy()
    Y_train = data_cut['I'].copy()

    poss = np.array([np.dot(Y_train, X_train[f]) / np.linalg.norm(X_train[f]) for f in features]).argmax()

    I_theor = np.zeros_like(Y_train)
    for i in np.arange(len(features)):
        I_theor += X_train[features[i]] * (1 - alpha)**i
    I_theor = np.array(I_theor)

    fig, ax = plt.subplots()
    legend = ['I', '$ I_{теор} $', '$ \Delta R $', tex_features[poss]]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    ax.plot(data_cut['X'], scaler.fit_transform(data_cut[['I']]))
    ax.plot(data_cut['X'], scaler.fit_transform(I_theor.reshape(-1, 1)))
    ax.plot(data_cut['X'], scaler.fit_transform(data_cut[['dR']]))
    ax.plot(data_cut['X'], scaler.fit_transform(delta[[features[poss]]]))

    ax.legend(legend)
    plt.title(bdp.country)
    plt.savefig('results/Belarus_IdRdX.pdf')


bdp = BelarusDataProcessor()
bdp.read_data(with_smooth=True)

#bdp.build_regression(with_constrains=True, values=np.array([40]))
#bdp.build_double_regression(with_constrains=False)

bel_plot_IdRDx()
