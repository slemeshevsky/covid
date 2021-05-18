# -*- coding: utf-8 -*-

#from IPython.core.pylabtools import figsize
from covid.utils import read
from covid.preprocess import smooth_average

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd
import patsy as pt
import sklearn.linear_model as lm


plt.ioff()

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
    fig, ax = plt.subplots()
    sns.lineplot(x=x_train.index, y=y_train, ax=ax, label='Сглаженные данные')
    sns.lineplot(x=x_train.index, y=y_pred, ax=ax, label='Регрессия')
    sns.lineplot(x=x_train.index, y=y_pred_1, ax=ax, label='Первая группа коэффициентов')
    sns.lineplot(x=x_train.index, y=y_pred_2, ax=ax, label='Вторая группа коэффициентов')

    fig.suptitle('Распределение инфицированных для окна {}'.format(len(model.coef_)))
    plt.savefig('results/Belarus_infected_{}'.format(len(model.coef_)))
    plt.close()

path = '../COVID-19/'

country = 'Belarus'
name = 'weeks'
left = 0
right = 70000
data, country = read(country, with_recovered=True, path = path)
data_sm = smooth_average(data, 7)

values = np.arange(35, 45)
delta = data_sm[['X','dX']].loc[(data_sm['X']>left) & (data_sm['X'] < right)].copy()
features, tex_features = calc_deltas(delta, values)

# Построение графиков коэффициентов с косинусом угла между I и \Delta X.
X_train = delta[features[::-1]].copy()
Y_train = (data_sm['X'] - data_sm['R']).loc[(data_sm['X']>left) & (data_sm['X'] < right)].copy()
model = lm.LinearRegression(fit_intercept=False, positive=True)
for wnd in values:
    model.fit(X_train[features[:wnd]], Y_train)
    scal  = np.array([np.dot(Y_train, X_train[f]) for f in features[:wnd]])
    norms = np.array([np.linalg.norm(Y_train) * np.linalg.norm(X_train[f]) for f in features[:wnd]])
    coss  = scal / norms
    plot_weights(tex_features[:wnd], model.coef_, coss, show_fig=False, sorting=False)

for wnd in values:
    model.fit(X_train[features[:wnd]], y_train)
    plot_infected(model, X_train[features[:wnd]], y_train, 27)


# tags_regr = []
# for val in values:
#     tags_regr.append('$ I_{' + str(val) + '} $')
# coefs = pd.DataFrame(np.zeros((values.max(), values.size)), index=range(1, values.max()+1), columns=tags_regr)

# fig, ax = plt.subplots()
# legend = ['I']
# ax.plot(deltas['X'], deltas['I'])
# # Plotting regression from X.
# skm = lm.LinearRegression(fit_intercept=False, positive=True)
# for n in values:
#     tag = '$ I_{' + str(n) + '} $'
#     x = pd.DataFrame(deltas, columns=[*tags_deltas[:n]]).to_numpy()
#     #x = pd.DataFrame(deltas, columns=[*reversed(tags_deltas[:n])]).to_numpy() # - for reversed data vectors.
#     y = deltas['I'].values
#     skm.fit(x, y)
#     result = np.zeros_like(y)
#     for i in range(n):
#         result += skm.coef_[i]*x[:,i]
#     coefs[tag][:n] = skm.coef_
#     #coefs[tag][-n:] = skm.coef_ # - shift reversed values.
#     deltas[tag] = result

#     legend.append(tag)
#     ax.plot(deltas['X'], deltas[tag], linewidth=1)
# ax.legend(legend)
# plt.title(country)
# plt.savefig('results/{0}_{1}_regrIfromX.pdf'.format(country, name))
# # Plotting regression from time.
# fig, ax = plt.subplots()
# legend = ['I']
# ax.plot(deltas.index, deltas['I'])
# for n in values:
#     tag = '$ I_{' + str(n) + '} $'
#     legend.append(tag)
#     ax.plot(deltas.index, deltas[tag], linewidth=1)
# ax.legend(legend)
# plt.title(country)
# plt.xticks(rotation=90)
# plt.subplots_adjust(bottom=0.2)
# plt.savefig('results/{0}_{1}_regrIfromT.pdf'.format(country, name))
# # Plotting coefficients.
# fig, ax = plt.subplots()
# for tag in tags_regr:
#     ax.plot(coefs.index, coefs[tag])
# ax.legend(tags_regr)
# plt.title('Coefficients of regression')
# plt.savefig('results/{0}_{1}_regrCoeffs.pdf'.format(country, name))

    model.fit(X_train[features[:wnd]], Y_train)
    plot_infected(model, X_train[features[:wnd]], Y_train, 25)

tags_regr = []
for val in values:
    tags_regr.append('$ I_{' + str(val) + '} $')
coefs = pd.DataFrame(np.zeros((values.max(), values.size)), index=range(1, values.max()+1), columns=tags_regr)

# Построение совместного графика I, \Delta R и \Delta X.
data_sm['I'] = data_sm['X'] - data_sm['R']
data_cut = data_sm[(data_sm['X'] > left) & (data_sm['X'] < right)].copy()
poss = np.array([np.dot(Y_train, X_train[f]) / np.linalg.norm(X_train[f]) for f in features]).argmax()

fig, ax = plt.subplots()
legend = ['I', '$ \Delta R $', '$ \Delta X_{' + str(poss) + '} $']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
ax.plot(data_cut['X'], scaler.fit_transform(data_cut[['I']]))
ax.plot(data_cut['X'], scaler.fit_transform(data_cut[['dR']]))
ax.plot(delta['X'], scaler.fit_transform(delta[[features[poss]]]))

ax.legend(legend)
plt.title(country)
plt.savefig('results/Belarus_IdRdX.pdf')

# Построение регрессии для возростающей и убывающей частей первой волны.
fig, ax = plt.subplots()
legend = ['I', 'left regression', 'right regression']
ax.plot(data_cut['X'], data_cut['I'])

border = data_cut.index[data_cut['I'].argmax()]
frames = [data_cut[data_cut.index <= border],
          data_cut[data_cut.index >= border]]

for frame in frames:
    model = lm.LinearRegression(fit_intercept=False, positive=True)
    X_train = delta[features[::-1]].loc[frame.index].copy()[features[:values.max()]]
    Y_train = data_sm['I'].loc[frame.index].copy()
    model.fit(X_train, Y_train)
    ax.plot(frame['X'], model.predict(X_train))

ax.legend(legend)
plt.title(country)
plt.savefig('results/Belarus_left-right_regression.pdf')
plt.close()
