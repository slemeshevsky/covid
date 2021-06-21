# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_s_w(data, cols=['X'], plot_sk=False, spacing=.2):
    colors = getattr(getattr(pd.plotting, '_matplotlib').style, '_get_standard_colors')(num_colors=len(cols)+2)
    ax = data[cols[0]].plot(label=cols[0], color=colors[0])
    ax.set_ylabel(ylabel=cols[0])
    lines, labels = ax.get_legend_handles_labels()
    for n in range(1, len(cols)):
        ax_n = ax.twinx()
        ax_n.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data[cols[n]].plot(ax=ax_n, label=cols[n], color=colors[n % len(colors)])
        ax_n.set_ylabel(ylabel=cols[n])
        line, label = ax_n.get_legend_handles_labels()
        lines += line
        labels += label

    ax_new = ax.twinx()
    ax_new.spines['right'].set_position(('axes', 1))
    if plot_sk:
        data[['S_k']].plot(ax=ax_new, label=['S'], color=colors[-1])
    data[['W']].plot(ax=ax_new, label=['W'], color=colors[-2])
    ax_new.set_ylabel(ylabel='$S, W$')
    ax_new.grid(linestyle=':')
    line, label = ax_new.get_legend_handles_labels()
    # lines += line
    # labels += label
    ax.legend(lines, labels, loc=0)
    return ax


def plot_multi(data, cols=None, spacing=0.05, **kwargs):
    from pandas import plotting

    # Get default color style from pandas - can be changed to any other color list
    if cols is None:
        cols = data.columns
    if len(cols) == 0:
        return
    colors = getattr(getattr(plotting, '_matplotlib').style, '_get_standard_colors')(num_colors=len(cols))

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0])
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs)
        ax_new.set_ylabel(ylabel=cols[n])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    return ax


def plot_weights(country, features, weights, norms, show_fig=True, sorting=True):
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

    sns.barplot(x=norms, y=features, ax=axs[1])
    axs[1].set_title(r"$ \frac{(I, \delta_k)}{||I|| \cdot ||\delta_k||} $")
    
    fig.suptitle('Окно = {}'.format(len(features)))
    plt.savefig('results/{0}_regrCoeffs_wind_{1}.pdf'.format(country, len(features)))
    if show_fig:
        plt.show()
    plt.close()


def plot_infected(country, model, x_train, y_train, n, x):
    y_pred = model.predict(x_train)
    coef_first = model.coef_.copy()
    coef_second = model.coef_.copy()
    coef_first[n:] = 0.0
    coef_second[:n] = 0.0
    y_pred_1 = np.dot(coef_first, x_train.T)
    y_pred_2 = np.dot(coef_second, x_train.T)
    fig, ax = plt.subplots()
    sns.lineplot(x=x, y=y_train, ax=ax, label='Сглаженные данные') # по X
    sns.lineplot(x=x, y=y_pred, ax=ax, label='Регрессия') # по X
    #sns.lineplot(x=x_train.index, y=y_train, ax=ax, label='Сглаженные данные') # по t
    #sns.lineplot(x=x_train.index, y=y_pred, ax=ax, label='Регрессия') # по t

    #sns.lineplot(x=x_train.index, y=y_pred_1, ax=ax, label='Первая группа коэффициентов')
    #sns.lineplot(x=x_train.index, y=y_pred_2, ax=ax, label='Вторая группа коэффициентов')
    
    plt.xticks(rotation=20)
    fig.suptitle('Распределение инфицированных для окна {}'.format(len(model.coef_)))
    plt.savefig('results/{0}_infected_{1}.pdf'.format(country, len(model.coef_)))
    plt.close()
