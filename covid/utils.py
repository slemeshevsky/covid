# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def get_country_name(country='Belarus', province='--'):
    return country if province == '--' else country + '(' + province + ')'


def fix_data(data, country):
    """
    Осуществляет исправление неполных или некорректных эпидемиологические данных.

    Параметры
    _________
    data : pandas.DataFrame
        Эпидемиологические данные.
    country : string
        Наименование страны.
    """
    if country == 'Belarus':
        data['2020-04-18'] = data['2020-04-17'] + 518
        data['2020-04-19'] = data['2020-04-18'] + 510
        # data['2020-04-20'] = data['2020-04-20'] - (518 + 510)
    if country == 'Spain':
        data['2020-04-24'] = data['2020-04-24':]+10000


def get_data(df, country='Belarus', province='--'):
    """
    Осуществляет подготовку сырых эпидемиологических данных, получаемых из репозитория университета Джонса Хопкинса.

    Параметры
    _________
    df : pandas.DataFrame
        Сырые данные.
    country : string
        Наименование страны.
    province : string
        Наименование провинции.

    Возвращаемые значения
    _____________________
    country_res : string
        Полное наименование эпидемиологического региона.
    data : pandas.DataFrame
        Подготовленные эпидемиологические данные.
    """
    country_res = get_country_name(country, province)

    tmp = df[df['Country/Region'] == country]
    df_ = tmp[tmp['Province/State'].isna()] if province == '--' else tmp[tmp['Province/State'] == province]
    all_data = pd.DataFrame(df_.T[4:].values, 
                            index=pd.date_range(start=df.columns[4:][0], end=df.columns[4:][-1]), columns=['Total'])
    data = all_data['Total'][all_data['Total'] > 0]

    fix_data(data, country_res)

    data = pd.DataFrame(data.values, index=data.index, columns=['X'])
    return country_res, data


def recalculate_deltas(data):
    """
    Высчитывает прирост основных величин в эпидемиологических данных.

    Параметры
    _________
    data : pandas.DataFrame
        Данные для пересчёта.
    """
    for val in ['X', 'R', 'D']:
        if val in data.columns:
            delta = np.zeros_like(data[val].values, dtype='float64')
            delta[0] = data[val].values[0]
            delta[1:] = data[val].values[1:] - data[val].values[:-1]
            data['d' + val] = delta


def read(country, province=None, with_recovered=False, path = 'https://github.com/CSSEGISandData/COVID-19/raw/master/'):
    """
    Осуществляет чтение эпидемиологических данных из репозитория университета Джонса Хопкинса.

    Параметры
    _________
    country : string
        Наименование страны.
    province : string
        Наименование провинции.
    with_recovered : bool
        Флаг загрузки данных по числу выздоровевших.
    path : string
        Путь к репозиторию с данными.

    Возвращаемые значения
    _____________________
    data : pandas.DataFrame
        Эпидемиологические данные.
    country : string
        Полное наименование эпидемиологического региона.
    """
    province = province if province is not None else '--'
    df = pd.read_csv(path + 'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    country, data = get_data(df, country, province=province)
    if with_recovered:
        df = pd.read_csv(path + 'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
        country, recovered = get_data(df, country, province)
        data['R'] = recovered['X'].copy()

        df = pd.read_csv(path + 'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
        country, deaths = get_data(df, country, province)
        data['D'] = deaths['X'].copy()

    recalculate_deltas(data)
    cols = data.columns[data.dtypes.eq('object')]
    data[cols] = data[cols].astype(np.float64)
    return data, country


def generate_verhulst_data(left=50, right=50, alpha=0.2, beta=0.05, gamma=0.07):
    # end = end if end is not None else datetime.today()
    # index = pd.date_range(start=start, end=end)
    # alpha = 0.2
    # beta = 0.05
    # gamma = 0.07
    p = 1/10.  # np.exp(middle_date*gamma)
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
    data['dX'] = pd.Series(delta, index=t[1:])
    # data['dX'].fillna(method='bfill', inplace=True)
    return data, t


def clean_work_columns(data):
    cols = ['Delta_mean', 'X_mean', 'S_k', 'S_k_regr', 'X_mean_scaled',
            'Delta_smooth', 'X_smooth', 'X_pred', 'Delta_pred', 'W']

    for col in cols:
        if col in data.columns:
            del data[col]


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

def v_cos(x, y):    
    scal = np.dot(x, y)
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return scal / norm
