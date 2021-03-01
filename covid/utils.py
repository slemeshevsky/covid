# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def get_data(df, country='Belarus', province='--'):
    if len(country) == 0:
        print('No information on {}'.format(country))
    elif len(df[df['Country/Region'] == country]) > 1:
        if province == '--':
            country_res = country
        else:
            country_res = country + '(' + province + ')'
    else:
        country_res = country

    tmp = df[df['Country/Region'] == country]
    df_ = tmp[tmp['Province/State'].isna()] if province == '--' else tmp[tmp['Province/State'] == province]
    all_data = pd.DataFrame(df_.T[4:].values, index=pd.date_range(start=df.columns[4:][0], end=df.columns[4:][-1]), columns=['Total'])
    data = all_data['Total'][all_data['Total'] > 0]
    if country == 'Belarus':
         data['2020-04-18'] = data['2020-04-17'] + 518
         data['2020-04-19'] = data['2020-04-18'] + 510
         # data['2020-04-20'] = data['2020-04-20'] - (518 + 510)
    if country == 'Spain':
         data['2020-04-24'] = data['2020-04-24':]+10000

    delta = np.zeros_like(data.values, dtype='float64')
    delta[0] = data.values[0]
    delta[1:] = data.values[1:] - data.values[:-1]

    data = pd.DataFrame(data.values, index=data.index, columns=['X'])
    data['Delta'] = delta

    return country_res, data

def read(country, province=None, with_recovered=False):
    province = province if province is not None else '--'
    df = pd.read_csv('https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    country, data = get_data(df, country, province=province) #'Belarus') # United Kingdom') # Belarus') # 'Belarus') # 'Germany') # 'Belgium')
    if with_recovered:
        df = pd.read_csv('https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
        country, recovered = get_data(df, country, province)
        df = pd.read_csv('https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
        country, deaths = get_data(df, country, province)
        data['Recovered'] = recovered['X'].copy()
        data['dR'] = recovered['Delta'].copy()
        data['Deaths'] = deaths['X'].copy()
        data['dD'] = deaths['Delta'].copy()
    return data, country

def generate_verhulst_data(left=50, right=50, alpha=0.2, beta=0.05, gamma=0.07):
    # end = end if end is not None else datetime.today()
    # index = pd.date_range(start=start, end=end)
    # alpha = 0.2
    # beta = 0.05
    # gamma = 0.07
    p = 1/10. # np.exp(middle_date*gamma)
    q = (1-p)/2.
    r = (1-p)/2.
    # p = 2.
    # right = len(index) - middle_date
    t = np.arange(-left, right)
    x = 1./(1. + 2*(q*np.exp(-alpha*t) + r*np.exp(-beta*t) + p*np.exp(-gamma*t)))
    data = pd.DataFrame(x, index=t, columns=['X'])
    # delta = np.zeros(len(x))
    # delta[0] = x[0]
    delta = x[1:] - x[:-1]
    data['Delta'] = pd.Series(delta, index=t[1:])
    # data['Delta'].fillna(method='bfill', inplace=True)
    return data, t

def clean_work_columns(data):
    cols = ['Delta_mean', 'X_mean', 'S_k', 'S_k_regr', 'X_mean_scaled',
            'Delta_smooth', 'X_smooth', 'X_pred', 'Delta_pred', 'W']

    for col in cols:
        if col in data.columns:
            del data[col]

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
    if cols is None: cols = data.columns
    if len(cols) == 0: return
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

