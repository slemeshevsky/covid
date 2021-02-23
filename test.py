# -*- coding: utf-8 -*-

import numpy as np
from covid.exponent_model import ExponentModel
from covid.utils import read
from covid.preprocess import smooth_q

# from covid.delay_model import ImplicitDelayEquation


def exponent(data):
    """Работа с экспоненциальной моделью

Parameters
----------
data : pd.DataFrame
    Данные для обучения

Returns
-------
out : covid.ExponentModel
    Модель
"""
    start_date = '2020-05-15'
    end_date = '2020-06-25'
    smooth_q(data)
    d_s = data['Delta_smooth'].rolling(7).mean()
    data = data.assign(Delta_7d=d_s)
    data = data.assign(X_7d=d_s.cumsum())
    X = data['X_7d'].loc[start_date:end_date].to_numpy()
    out = {}

    for p in np.arange(int(len(X)/3), 0, -1):
        try:
            exp_mdl = ExponentModel(p=p,
                                    init_date=start_date,
                                    number_of_exp=3).fit(X)
            out[p] = exp_mdl
        except ValueError:
            continue

    return out


data, country = read('Belarus')
start_date = '2020-05-01'

work_data = data.loc[:'2020-07-30']
models = exponent(work_data)
