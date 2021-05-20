# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import root


def smooth_q(data, eps=0.125e-2):
    """
    Осуществляет сглаживание левого склона.

    Параметры
    _________
    data : pandas.DataFrame
        Данные для сглаживания.
    eps : float
        Зазор для срезания

    Возвращаемые значения
    _____________________
    delta : array
        Массив сглаженных ежедневных приростов
    x : array
        Массив сглаженных текущих накопленных значений


    Замечания
    _________
    Сглаженные данные также дописываются во входной data
    """
    delta = data['dX'].to_numpy().copy()
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
    return delta, x


def smooth_s(data, coef):
    """
    Осуществляет сглаживание остальных данных на основе
    уравнения с запаздыванием.

    Параметры
    _________
    data : pandas.DataFrame
        Данные для сглаживания
    coef : список размерности 2
        Коэффициенты модели на основе уравнения с
        запаздыванием

    Возвращаемые значения
    _____________________
    delta : array
        Массив сглаженных ежедневных приростов
    x : array
        Массив сглаженных текущих накопленных значений
    Замечания
    _________
    Сглаженные данные также дописываются во входной data
    """
    delta = data['Delta_smooth'].to_numpy().copy()
    x = data['X_smooth'].to_numpy(dtype='float64').copy()
    s = coef[1] + coef[0]*data['X_smooth'].to_numpy()
    # s = data['S_k_regr'].to_numpy().copy()
    q = np.zeros_like(s)

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
    return q, s


def smooth_average(data, period):
    """
    Осуществляет сглаживание данных по среднему значению за период. 
    После перестраивает значение дельт.

    Параметры
    _________
    data : pandas.DataFrame
        Данные для сглаживания.
    period : int
        Размер интервала сглаживания.

    Возвращаемые значения
    _____________________
    smooth_data : pandas.DataFrame
        Массив сглаженных значений.
    """
    from covid.utils import recalculate_deltas
    smooth_data = data.rolling(period).mean().dropna()
    recalculate_deltas(smooth_data)
    return smooth_data



def smooth_average_deltas(data, period):
    """
    Осуществляет сглаживание дельт данных по среднему значению за период.
    После перестраивает значения данных по новым дельтам.

    Параметры
    _________
    data : pandas.DataFrame
        Данные для сглаживания.
    period : int
        Размер интервала сглаживания.

    Возвращаемые значения
    _____________________
    smooth_data : pandas.DataFrame
        Массив сглаженных значений.
    """
    from covid.utils import recalculate_deltas
    smooth_data = data.rolling(period).mean().dropna()
    for val in ['X', 'R', 'D']:
        if val in data.columns:
            col = np.zeros_like(data[val].values, dtype='float64')
            col[0] = data[val].values[0]
            col[1:] = data[val].values[:-1] + data['d' + val].values[1:]
            data[val] = col
    return smooth_data
