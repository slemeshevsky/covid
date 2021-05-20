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
