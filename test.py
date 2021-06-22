# -*- coding: utf-8 -*-

from IPython.core.pylabtools import figsize
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
    alpha = 0.02

    cdp, country_info = get_country_data_processor('Belarus')
    cdp.read_data(with_smooth=True)

    data_cut = cdp.get_wave()
    delta, features, tex_features = cdp.calc_deltas(num, data_cut.index)

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
    plt.title(cdp.country)
    plt.savefig('results/Belarus_IdRdX.pdf')


def calc_regression(X, p):
    result = np.zeros_like(X[:, 0])
    for i in range(p.size):
        result += X[:, i] * p[i]
    return result


def lin_appr(country='Belarus', num_wave=1, with_scale=False):
    # Построение аппроксимации инфецированных с помощью искуственных коэффициентов (линейных).
    N = 40

    cdp, country_info = get_country_data_processor(country)

    data_cut = cdp.get_wave(num_wave)
    deltas, features, tex_features = cdp.calc_deltas(N, data_cut.index)

    if with_scale:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

    for L in range(N+1):
        p = np.ones(N+1)
        p[L:] = (N+1 - np.arange(L, N+1)) / float(N+1-L)
        regr = calc_regression(deltas.to_numpy(), p)

        if with_scale:
            I = scaler.fit_transform(data_cut[['I']])
            regression = scaler.fit_transform(regr.reshape(-1, 1))
        else:
            I = data_cut['I']
            regression = regr

        fig, ax = plt.subplots()
        legend = ['I', 'Regression']
        ax.plot(data_cut.index, I)
        ax.plot(data_cut.index, regression)
        ax.legend(legend) 
        plt.title('{0}, L = {1}'.format(country, L))
        plt.savefig('results/{0}_lin_regression_L{1}{2}.pdf'.format(country, L, ('_scaled' if with_scale else '')))
        plt.close()


def geom_appr(country='Belarus', num_wave=1, with_scale=False):
    # Построение аппроксимации инфецированных с помощью искуственных коэффициентов (геометрических).
    N = 40

    h = 0.1
    q = np.arange(h, 1, h)

    cdp, country_info = get_country_data_processor(country)

    data_cut = cdp.get_wave(num_wave)
    deltas, features, tex_features = cdp.calc_deltas(N, data_cut.index)

    if with_scale:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

    for L in range(N+1):
        regr = np.zeros([len(q), len(data_cut.index)])
        p = np.ones([len(q), N+1])
        for i in range(len(q)):
            p[i, L:] = q[i] ** np.arange(1, N-L+2)
            regr[i] = calc_regression(deltas.to_numpy(), p[i])

        fig, ax = plt.subplots()
        legend = ['I']
        if with_scale:
            I = scaler.fit_transform(data_cut[['I']])
        else:
            I = data_cut['I']
        ax.plot(data_cut['X'], I)

        for i in range(len(q)):
            legend.append('q = {:1.2f}'.format(q[i]))
            if with_scale:
                regression = scaler.fit_transform(regr[i].reshape(-1, 1))
            else:
                regression = regr[i]
            ax.plot(data_cut['X'], regression)

        ax.legend(legend)
        plt.title('{0}, L = {1}'.format(country, L))
        plt.savefig('results/{0}_geom_regression_L{1}{2}.pdf'.format(country, L, ('_scaled' if with_scale else '')))
        plt.close()


#cdp, country_info = get_country_data_processor('Belarus')
#cdp.read_data(with_smooth=True)
#cdp.build_regression(with_constrains=True, values=np.array([40]))
#cdp.build_double_regression(num=country_info['double_regression_max_num'], with_constrains=True)

#for with_scale in [False, True]:
#    lin_appr(with_scale=with_scale)
#    geom_appr(with_scale=with_scale)


cdp, country_info = get_country_data_processor('Belarus')
cdp.read_data(with_smooth=True)
num = 40

data_cut = cdp.get_wave()
delta, features, tex_features = cdp.calc_deltas(num)

X_train = delta[features[::-1]].loc[data_cut.index].copy()
Y_train = data_cut['I'].copy()

from covid.ConstrainedLinearRegression import ConstrainedLinearRegression
from covid.utils import v_cos
model = ConstrainedLinearRegression()
model.fit(X_train[features[:num]], Y_train, min_coef=np.zeros(num), max_coef=np.ones(num))

print(model.coef_)
print(model.intercept_)

model.coef_ = np.ones_like(model.coef_)
coss = np.array([v_cos(Y_train, X_train[f]) for f in features[:num]])

y_model = model.predict(X_train[features[::-1]])
y_test = calc_regression(X_train[features[::-1]].to_numpy(), model.coef_)

fig, ax = plt.subplots()
sns.lineplot(x=data_cut.index,
             y=data_cut['I'],
             ax=ax,
             label='Сглаженные данные')
sns.lineplot(x=data_cut.index,
             y=y_model,
             ax=ax,
             label='Регрессия')
sns.lineplot(x=data_cut.index,
             y=y_test,
             ax=ax,
             label='Регрессия (тест функции)')

plt.xticks(rotation=20)

norm = np.linalg.norm(y_model - y_test)
fig.suptitle('Норма разницы построенных регрессий {}'.format(norm))
plt.savefig('results/{0}_infected_{1}.pdf'.format(cdp.country, num))
plt.close()
