# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import root
from sklearn.linear_model import LinearRegression


def smooth_q(data, eps=0.125e-2):
    delta = data['Delta'].to_numpy().copy()
    x = data['X'].to_numpy(dtype='float64').copy()
    for col in ['Delta_smooth', 'X_smooth']:
        if col in data.columns:
            del data[col]
    for w in [7, 5]:
        q = np.zeros_like(delta)
        s = np.zeros_like(q)
        for k in range(w+1, len(q)):
            q[k] = (x[k-1]/x[k-w-1])**(1/w)
            q1 = q[k] + eps
            s[k] = (q1-1)/(1-q1**(-w-1))
            Delta = x[k-1] - x[k-w-1]
            R = delta[k] - s[k]*Delta
            if R > 0:
                b = R*(q1-1)/(q1**(w)-1)
                delta[k] = s[k]*Delta
                p = [i-(k-w) for i in range(k-w, k)]
                delta[k-w:k] += b*q1**p
                x = delta.cumsum()
    data['Delta_smooth'] = delta.copy()
    data['X_smooth'] = x.copy()
    return q, s


def w_k(data, col=None, l=7):
    X_col = 'X_'+col if col is not None else 'X'
    x = data[X_col].values.astype('float64') if col is not None \
        else data[data[X_col] > 0][X_col].values.astype('float64')
    index = data.index if col is not None else data[data[X_col] > 0].index
    s = (x[l:] - x[l-1:-1])/(x[l:]-x[:-l])
    # s = np.zeros(len(x)-l)
    # for k in range(len(s)):
    #     s[k] = (x[k+l] - x[k+l-1])/(x[k+l-1] - x[k])
    if 'W' in data.columns:
        del data['W']
    data['W'] = pd.Series(s, index=index[l:])
    return pd.Series(s, index=index[l:])


def s_k(data, col=None, l=7):
    X_col = 'X_'+col if col is not None else 'X'
    x = data[X_col].values.astype('float64') if col is not None \
        else data[data[X_col] > 0][X_col].values.astype('float64')
    index = data.index if col is not None else data[data[X_col] > 0].index
    s = (x[l:] - x[l-1:-1])/(x[l-1:-1]-x[:-l])
    # s = np.zeros(len(x)-l)
    # for k in range(len(s)):
    #     s[k] = (x[k+l] - x[k+l-1])/(x[k+l-1] - x[k])
    if 'S_k' in data.columns:
        del data['S_k']
    data['S_k'] = pd.Series(s, index=index[l:])
    return pd.Series(s, index=index[l:])


def delay_explicit(data, use_regr=False, coef=None, start=None, end=None, l=7):
    if use_regr and (coef is None):
        raise ValueError('Необходимо задать коэффициенты регрессии')

    start = start if start is not None else data.index[l]
    end = end if end is not None else data.index[-1]
    time = pd.date_range(start=start, end=end)
    t = np.arange(len(time))
    k0 = data.index.get_loc(start)
    col_id = 'S_k_regr' if use_regr else 'S_k'
    x_pred = np.zeros(len(t))
    x_pred[:l] = data['X_smooth'].values[k0:k0+l].copy()
    for k in range(l, len(x_pred)):
        if coef is not None:
            s = coef[1] + coef[0]*x_pred[k-1]
        else:
            s = data[col_id].values[k0+k]
        x_pred[k] = x_pred[k-1] + s*(x_pred[k-1] - x_pred[k-l])
    delta_pred = x_pred[1:] - x_pred[:-1]

    result = pd.DataFrame(x_pred, index=time, columns=['X'])
    result['Delta'] = pd.Series(delta_pred, index=time[1:])
    if 'X_pred' in data.columns:
        del data['X_pred']
    if 'Delta_pred' in data.columns:
        del data['Delta_pred']
    if (start == data.index[0]) and (end == data.index[-1]):
        data['X_pred'] = pd.Series(x_pred, index=data.index[k0:])
        data['Delta_pred'] = pd.Series(delta_pred, index=data.index[k0+1:])
    return result

def delay_eq_expl(init, r, K, start=None, end=None, l=7):
    t = pd.date_range(start=start, end=end)
    x_pred = np.zeros(len(t))
    x_pred[:len(init)] = init
    for k in range(l, len(x_pred)):
        s = r*K - r*x_pred[k-1]
        x_pred[k] = x_pred[k-1] + s*(x_pred[k-1] - x_pred[k-l])

    delta_pred = x_pred[1:] - x_pred[:-1]
    result = pd.DataFrame(x_pred, index=t, columns=['X'])
    result['Delta'] = pd.Series(delta_pred, index=t[1:])

    return pd.DataFrame(x_pred, index=t)


def delay_implicit(data, use_regr=False, coef=None, start=None, end=None, init=None, l=7):
    if use_regr and (coef is None):
        raise ValueError('Необходимо задать коэффициенты регрессии')

    init = init if init is not None else data.index[l]
    start = start if start is not None else init
    end = end if end is not None else data.index[-1]
    time = pd.date_range(start=start, end=end)
    left = len(time[time < init])
    right = len(time) - left
    t = np.arange(-left, right)
    k0 = data.index.get_loc(init)
    # print(k0)
    col_id = 'W_regr' if use_regr else 'W'
    x_pred = np.zeros(len(t))
    x_pred[left:left+l] = data.iloc[k0:k0+l]['X_smooth'].to_numpy(dtype='float64')
    for k in range(left+l, len(x_pred)):
        if coef is not None:
            s = coef[1] + coef[0]*x_pred[k-1]
        else:
            s = data[col_id].values[k0+k]
        x_pred[k] = (x_pred[k-1] - s*x_pred[k-l])/(1-s)

    # if left > l:
    #     for k in range(left+l-1, l-1, -1):
    #         if coef is not None:
    #             s = coef[1] + coef[0]*x_pred[k-1]
    #         else:
    #             s = data[col_id].values[k0+k]
    #         x_pred[k-l] = (x_pred[k-1] - (1-s)*x_pred[k])/s
    delta_pred = x_pred[1:] - x_pred[:-1]

    result = pd.DataFrame(x_pred, index=time, columns=['X'])
    result['Delta'] = pd.Series(delta_pred, index=time[1:])
    if 'X_pred' in data.columns:
        del data['X_pred']
    if 'Delta_pred' in data.columns:
        del data['Delta_pred']
    if (start == data.index[0]) and (end == data.index[-1]):
        data['X_pred'] = pd.Series(x_pred, index=data.index[k0:])
        data['Delta_pred'] = pd.Series(delta_pred, index=data.index[k0+1:])
    return result


def fit_sk(data, col=None, days=7, left=7, right=-8):
    for c in ['S_k_regr']:
        if c in data.columns:
            del data[c]
    if col is not None:
        X_col = 'X_'+col
        Delta_col = 'Delta_'+col
    else:
        X_col = 'X'
        Delta_col = 'Delta'

    # left = len(data) - left
    s_k(data, col=col, l=days)
    # left = -left
    left += data[Delta_col].argmax()
    right = right
    x_model = data.iloc[left:right][X_col].to_numpy(dtype='float64')
    y = data.iloc[left:right]['S_k'].to_numpy()
    model = LinearRegression().fit(x_model[:, None], y)
    score = model.score(x_model[:, None], y)
    a = model.coef_[0]
    b = model.intercept_
    # x = data[X_col].values[:-1]
    # s_regr = b + a*x
    data['S_k_regr'] = b + a*data[X_col] # pd.Series(s_regr, index=data.index[1:])
    return [a, b], score


def fit_w(data, col=None, days=7, left=7, right=-8):
    for c in ['W_regr']:
        if c in data.columns:
            del data[c]
    if col is not None:
        X_col = 'X_'+col
        Delta_col = 'Delta_'+col
    else:
        X_col = 'X'
        Delta_col = 'Delta'

    w_k(data, col=col, l=days)
    left += data[Delta_col].argmax()
    right = right
    x_model = data.iloc[left:right][X_col].to_numpy(dtype='float64')
    y = data.iloc[left:right]['W'].to_numpy()
    model = LinearRegression().fit(x_model[:, None], y)
    score = model.score(x_model[:, None], y)
    a = model.coef_[0]
    b = model.intercept_
    data['W_regr'] = b + a*data[X_col]
    return [a, b], score


def smooth_s(data, coef):
    delta = data['Delta_smooth'].to_numpy().copy()
    x = data['X_smooth'].to_numpy(dtype='float64').copy()
    s = coef[1] + coef[0]*data['X_smooth'].to_numpy()
    # s = data['S_k_regr'].to_numpy().copy()
    q = np.zeros_like(s)
#    start = 0
    for w in [7, 5]:
        for k in range(w+1, len(s)):
            sol = root(lambda x: x**(w+1) - s[k]*((x**(w+1)-1)/(x-1)), s[k]+1)
            q[k] = sol.x
            Delta = x[k-1] - x[k-w-1]
            R = delta[k] - s[k]*Delta
            if R > 0:
                delta[k] = s[k]*Delta
                p = np.array([i-(k-w) for i in range(k-w, k)])
                b = R*(q[k-w-1]-1)/(q[k-w-1]**(w)-1)
                delta[k-w:k] += b*q[k-w-1]**p
            x = delta.cumsum()
    data['Delta_smooth'] = delta.copy()
    data['X_smooth'] = x.copy()
    return q, s, x


def smooth_w(data, coef):
    delta = data['Delta_smooth'].to_numpy().copy()
    x = data['X_smooth'].to_numpy(dtype='float64').copy()
    s = coef[1] + coef[0]*data['X_smooth'].to_numpy()
    # s = data['S_k'].fillna(0.0).to_numpy().copy()
    q = np.zeros_like(s)
    start = 0
    for w in [7, 5]:
        for k in range(start+w+1, len(s)):
            sol = root(lambda x: x**(w+1) - s[k]*((x**(w+1)-1)/(x-1)), s[k]+1)
            q[k] = sol.x
            Delta = x[k-1] - x[k-w-1]
            R = delta[k] - s[k]*Delta
            if R > 0:
                delta[k] = s[k]*Delta
                p = np.array([i-(k-w) for i in range(k-w, k)])
                b = R*(q[k-w-1]-1)/(q[k-w-1]**(w)-1)
                delta[k-w:k] += b*q[k-w-1]**p
            x = delta.cumsum()
    data['Delta_smooth'] = delta.copy()
    data['X_smooth'] = x.copy()
    return q, s


def delay_predict(data, country, width, l=7, date=None,
                  pred_start=None, pred_end='2020-09-30',
                  smooth_left=False, explicit=False):
    params = {'Country': country}
    q, s = smooth_q(data)
    if smooth_left:
        if explicit:
            coef, score = fit_sk(fit_data, col='smooth', days=l)
            q, s = smooth_s(fit_data,  coef)
        else:
            coef, score = fit_w(fit_data, col='smooth', days=l, start=init, end=date)
            q, s = smooth_w(fit_data,  coef)
    date = date if date is not None else data.index[-1]
    fit_data = data[data.index < date].copy()
    init = fit_data.index[-width]
    # pred_start = pred_start if pred_start is not None
    if explicit:
        coef, score = fit_sk(fit_data, col='smooth', days=l, left=width, right=-8)
        pred = delay_explicit(fit_data, end=pred_end, init=init, use_regr=True, coef=coef)
    else:
        coef, score = fit_w(fit_data, col='smooth', days=l, start=init, end=date)
        pred = delay_implicit(fit_data, end=pred_end, init=init, use_regr=True, coef=coef)
    params['r'] = -coef[0]
    params['K'] = -coef[1]/coef[0]
    params['sup'] = pred['X'].max()
    params['score'] = score
    params['date'] = date
    params['width'] = width
    params['l'] = l
    return params, pred


def main():
    clean_work_columns(data)
    params_expl = {'Country': country}
    # data['Delta_mean'] = data.Delta.rolling(7).mean()
    # data['Delta_mean'] = data.Delta_mean.rolling(5).mean()
    # data['Delta_mean'] = data.Delta_mean.rolling(3).mean()
    # data['X_mean'] = data['Delta_mean'].cumsum()
    start = '2020-05-01'
    end = '2020-08-31'
    q, s = smooth_q(data)
    coef, score = fit_sk(data, col='smooth', left=13, right=-8)
    params_expl['r'] = -coef[0]
    params_expl['sup'] = -coef[1]/coef[0]
    params_expl['score'] = score
    pred_expl = delay_explicit(data, start=start, end=end, use_regr=True, coef=coef)
    params_impl = {'Country': country}
    coef, score = fit_w(data, col='smooth', left=13, right=-8)
    params_impl['r'] = -coef[0]
    params_impl['sup'] = -coef[1]/coef[0]
    params_impl['score'] = score
    pred_impl = delay_implicit(data, start=start, end=end, use_regr=True, coef=coef)
    return params_expl, pred_expl, params_impl, pred_impl


def source_data_plot(country, province=None, date=None, smooth_left=False):
    data, country = read(country, province=province)
    date = date if date is not None else data.index[-1]
    q, s = smooth_q(data)
    data['Delta_mean'] = data.Delta.rolling(7).mean()
    data['X_mean'] = data['Delta_mean'].cumsum()
    if smooth_left:
        # coef, score = fit_sk(data, col='mean', left=10)
        # q, s = smooth_s(data, coef)
        coef, score = fit_sk(data, col='smooth', left=10)
        q, s = smooth_s(data, coef)
    data[data.index<date]['Delta'].plot(label='Исходные данные')
    data[data.index<date]['Delta_smooth'].plot(label='Сглаженные данные')
    data[data.index<date]['Delta_mean'].plot(label='Недельное среднее')
    plt.legend()
    plt.title(country)
    country = 'Hubei' if country == 'China(Hubei)' else country 
    plt.savefig('results/{}/3_delta.png'.format(country))
    return data, country


def get_K(data, country, dates):
    index = []
    K = []
    for date in dates:
        params, pred = delay_predict(data, country, 25, l=7, date=date)
        index.append(date)
        K.append(params['K'])
    return pd.Series(K, index=index)


def smooth(data):
    clean_work_columns(data)
    data['Delta_mean'] = data.Delta.rolling(7).mean()
    data['X_mean'] = data['Delta_mean'].cumsum()
    q, s = smooth_q(data)
    coef, score = fit_w(data, col='smooth', left=-10)
    q, s = smooth_w(data, coef)
    return score

if __name__ == '__main__':
    data, country = read('Belarus')
    dates = pd.date_range(start='2020-06-01', end='2020-07-07')
    widths = range(7, 20)
    generate_delay_report(data, country=country, widths=widths, dates=dates, method=delay_predict)


def gamma(country):
    from pandas import plotting
    confirmed, country = read(country)
    df = pd.read_csv('https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    country, recovered = get_data(df, country)
    recovered.rename(columns={'X': 'X_rec', 'Delta': 'Delta_rec'}, inplace=True)
    data = confirmed.join(recovered)
    data['sum'] = data['X'].cumsum()
    data['gamma'] = data['sum']/data['X_rec']

    ax = data['gamma'].plot(label='$\gamma_n$', color='b')
    ax.set_ylabel(ylabel='$\gamma_n$')
    lines, labels = ax.get_legend_handles_labels()

    ax_new = ax.twinx()
    ax_new.spines['right'].set_position(('axes', 1))
    data['Delta'].plot(ax=ax_new, label='Ежедневный прирост', color='y')
    ax_new.set_ylabel(ylabel='Ежедневный прирост')
    #ax_new.grid(linestyle=':')
    line, label = ax_new.get_legend_handles_labels()
    lines += line
    labels += label

    plt.title(country)
    ax.legend(lines, labels, loc=0)
    plt.savefig('results/{c}/gamma_{c}.png'.format(c=country))


def test_u(country):
    from pandas import plotting
    confirmed, country = read(country)
    df = pd.read_csv('https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    country, recovered = get_data(df, country)
    recovered.rename(columns={'X': 'X_rec', 'Delta': 'Delta_rec'}, inplace=True)
    data = confirmed.join(recovered)
    data['u'] = (data['Delta'] + data['Delta_rec'])/data['X']
    data['v'] = data['X'] + data['X_rec']
    data[['u', 'v']].plot(x='v')
    plt.title(country)
    plt.savefig('results/{c}/u_vs_v_{c}.png'.format(c=country))

    return data
