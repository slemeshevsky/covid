# -*- coding: utf-8 -*-
from covid.ConstrainedLinearRegression import ConstrainedLinearRegression
from covid.plotting import plot_weights, plot_infected
from covid.preprocess import smooth_average, smooth_average_deltas
from covid.utils import read, calc_deltas, v_cos

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.linear_model as lm

class CountryDataProcessor(object):
    """Класс для обработки эпидемиологических данных по выбранной стране.

Parameters
----------
data : Эпидемиологические данные.
country : Наименование страны.
wave : Границы волн.

    """


    def __init__(self, country, waves=[0, 0]):
        """
    Инициализирует объект
        """
        self.data = pd.DataFrame()
        self.country = country
        self.wave = waves

    
    def update_country_info(self, country):
        if country == 'Belarus':
            self.country = country
            self.waves = [0, 70000]
            self.double_regression_max_num = 64
            self.regression_values = np.arange(38, 43)
        else:
            self.country = ""
            self.waves = [0, 0]
            self.double_regression_max_num = 1
            self.regression_values = np.arange(0, 2)
            print("Haven't information about country {}.".format(country))


    def read_data(self, with_smooth=False, path='../COVID-19/'):
        """
    Функция для считывания данных.

    Parameters
    ----------
    with_smooth : bool
        Необходимость семидневного сглаживания исходных данных.
    path : string
        Путь к исходным данным.
        """
        self.data, self.country = read(self.country, with_recovered=True, path = path)
        self.data['I'] = self.data['X'] - self.data['R']
        if with_smooth:
            #self.data = smooth_average(self.data, 7).iloc[1:] # smooth -> reaclculate deltas with data
            self.data = smooth_average_deltas(self.data, 7).iloc[1:] # smooth -> recalculate data with deltas


    def get_wave(self, num=1):
        """
    Функция выборки данных конкретной волны.

    Parameters
    ----------
    num : int
        Номер волны.

    Returns
    -------

     wave_data : pandas.DataFrame
        Данные волны с номером num.
        """
        left = self.wave[num-1]
        right = self.wave[num]
        wave_data = self.data[(self.data['X'] > left) & (self.data['X'] < right)].copy()
        return wave_data


    def calc_deltas(self, num=1, index=None):
        """
    Вычисление дельт dX со сдвигом.

    Parameters
    ----------
    num : int
        Максимальная длина сдвига.
    index : pandas.Index
        Даты для вычислинения дельт.

    Returns
    -------

    delta : pandas.DataFrame
        Дельты X.
    features : list размерности num
    tex_features : list размерности num
        """
        if index is None:
            index = self.data.index
        delta = self.data[['X','dX']].loc[index].copy()
        features, tex_features = calc_deltas(delta, [num])
        return delta, features, tex_features


    def build_regression(self, values, wave=1, with_constrains=False):
        """
    Построение регрессии для выбранной волны заболевания.
    Построение коэффициентов регрессии с косинусом угла между I и \Delta X.

    Parameters
    ----------
    values : np.array
        Количество коэффициентов регрессии.
    wave : int
        Номер волны.
    with_constrains : bool
        Ограничение коэффициентов регрессии меньше единицы.
        """
        data_cut = self.get_wave()
        delta, features, tex_features = self.calc_deltas(values.max())

        X_train = delta[features[::-1]].loc[data_cut.index].copy()
        Y_train = data_cut['I'].copy()
        
        model = (ConstrainedLinearRegression() if with_constrains else lm.LinearRegression(fit_intercept=False, positive=True))
        for wnd in values:
            if with_constrains:
                model.fit(X_train[features[:wnd]], Y_train, min_coef=np.zeros(wnd), max_coef=np.ones(wnd))
            else:
                model.fit(X_train[features[:wnd]], Y_train)
            model.coef_ = np.ones(wnd)
            coss = np.array([v_cos(Y_train, X_train[f]) for f in features[:wnd]])
            plot_weights(self.country, tex_features[:wnd], model.coef_, coss, show_fig=False, sorting=False)
            plot_infected(self.country, model, X_train[features[:wnd]], Y_train, 25)


    def build_double_regression(self, num=10, wave=1, with_constrains=False):
        """
    Построение регрессии для возростающей и убывающей частей выбранной волны.

    Parameters
    ----------
    wave : int
        Номер волны.
    with_constrains : bool
        Ограничение коэффициентов регрессии меньше единицы.
        """
        data_cut = self.get_wave()
        delta, features, tex_features = self.calc_deltas(num, data_cut.index)
        border = data_cut.index[data_cut['I'].argmax()]

        fig, ax = plt.subplots()
        legend = ['I', 'left regression', 'right regression']
        ax.plot(data_cut['X'], data_cut['I'])

        model = (ConstrainedLinearRegression(fit_intercept=False) if with_constrains else lm.LinearRegression(fit_intercept=False, positive=True))

        frames = [{'name': 'left',  'data': data_cut[data_cut.index <= border]},
                  {'name': 'right', 'data': data_cut[data_cut.index >= border]}]
        for frame in frames:
            data = frame['data']
            index = data.index
            name = frame['name']

            X_train = delta[features[::-1]].loc[index].copy()
            Y_train = data['I'].copy()

            if with_constrains:
                model.fit(X_train, Y_train, min_coef=np.zeros(num), max_coef=np.ones(num))
            else:
                model.fit(X_train, Y_train)

            predict = model.predict(X_train.loc[index])
            ax.plot(data['X'], predict)

            diff = np.linalg.norm(predict - Y_train.to_numpy())
            coss = np.array([v_cos(Y_train, X_train[f]) for f in features[::-1]])

            fig, axs = plt.subplots(ncols=2)
            sns.barplot(y=tex_features[::-1], x=model.coef_, ax=axs[0])
            axs[0].set_xlabel("Веса")
            axs[0].set_title("Коэффициенты регрессии")
            sns.barplot(x=coss, y=tex_features[::-1], ax=axs[1])
            axs[1].set_title(r"$ \frac{(I, \delta_k)}{||I|| \cdot ||\delta_k||} $")
            fig.suptitle('Окно = {0} ({1})'.format(len(tex_features), diff))

            plt.savefig('results/Belarus_left-right_regrCoeffs_{}.pdf'.format(name))
            plt.close()


        ax.legend(legend) 
        plt.title(self.country)
        plt.savefig('results/Belarus_left-right_regression.pdf')
        plt.close()
