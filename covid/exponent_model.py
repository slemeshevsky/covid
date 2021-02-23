# -*- coding: utf-8 -*-

"""Модуль содержит класс для моделей, основанных на сумме экспонент

.. math::

   x(t) = \\frac{1}{a_0 + \sum_{k=1}^{m_e} a_k e^{\\alpha_k t}}

"""

from .base import BaseModel
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class ExponentModel(BaseModel):
    """Класс для моделей основанных на суммах экспонент

Parameters
----------
p : int
    Регуляризатор: ширина сдвига
number_of_exp : int
    Количество экспонент в модели (:math:`m`). По умолчанию 2
init_date : строка даты вида ``YYYY-MM-DD``
    Начальная дата для модели
    """

    def __init__(self, p, init_date, number_of_exp=2):
        """Инициализация объекта

Parameters
----------
p : int
    Регуляризатор: ширина сдвига
init_date : строка даты вида ``YYYY-MM-DD``
    Начальная дата для модели
number_of_exp : int
    Количество экспонент в модели (:math:`m`). По умолчанию ``2``
        """
        super().__init__()
        self.number_of_exp = number_of_exp
        self.c = np.zeros(self.number_of_exp)
        self.p = p
        self.intercept_ = 0.0
        self.init_date = init_date

    def __str__(self):
        s = """Экспоненциальная модель
\t Количество экспонент: {}
\t Коэффициенты: {}
\t Свободный член: {}
        """.format(self.number_of_exp, self.c, self.intercept_)
        return s

    def __repr__(self):
        s = """Экспоненциальная модель
\t Количество экспонент: {}
\t Ширина сдвига: {}
\t Коэффициенты: {}
\t Свободный член: {}
\t Качество: {}
        """.format(self.number_of_exp, self.p,  self.c, self.intercept_, self.score)
        return s

    def fit(self, X):
        """Обучение модели

Parameters
----------
X : array
    Данные для обучения.

Returns
-------
self : Экземпляр объекта.
        """
        n_samples = len(X) - self.number_of_exp*self.p
        Z = np.zeros((n_samples, self.number_of_exp+1))
        Z[:, -1] = 1/X[self.number_of_exp*self.p:]
        for k in range(self.number_of_exp):
            v = 1/X[k*self.p:]
            Z[:, k] = v[:(k-self.number_of_exp)*self.p]

        samples = Z[:, :-1]
        target = Z[:, -1]
        model = LinearRegression().fit(samples, target)
        self.intercept_ = model.intercept_
        poly = np.concatenate((np.ones(1), -model.coef_[::-1]))
        self.mu = np.roots(poly)
        if ((self.mu.dtype == np.dtype(np.complex)) or (self.mu <= 0.).any()):
            raise ValueError("Есть хотя бы один отрицательный либо комплексный корень!!!")
        self.mu = self.mu**(1/self.p)
        xi_train = np.array([self.mu**k for k in range(len(X))])
        z_train = 1/X
        mdl = LinearRegression().fit(xi_train, z_train)
        self.intercept_ = mdl.intercept_
        self.c = np.array(mdl.coef_)
        x_test = self._decision_func(np.arange(len(X)))

        self.score = mdl.score(xi_train, z_train)
        self.intercept_ = mdl.intercept_
        self.c = np.array(mdl.coef_)

        return self

    def _decision_func(self, t):
        """Функция, описывающая модель

Parameters
----------
t : array
    Моменты времени для вычисления прогноза

Returns
-------
array
    Значения прогноза
        """
        denom = self.intercept_
        for i in range(self.number_of_exp):
            denom += self.c[i]*self.mu[i]**t
        return 1/denom

    def predict(self, start, end):
        """Построение прогноза

Parameters
----------
start : строка даты вида ``YYYY-MM-DD``
    Начало прогноза
end : строка даты вида ``YYYY-MM-DD``
    Окончание прогноза

Returns
-------
res : pandas.DataFrame с columns=['X', 'Delta'] и
    index=pandas.date_range(start=start, end=end)
    Прогнозные значения
        """
        indices = pd.date_range(start=start, end=end)
        init = indices.get_loc(self.init_date)
        left = -init
        right = len(indices) - init
        t = np.arange(left, right)
        x = self._decision_func(t)
        res = pd.DataFrame(x, index=indices, columns=['X'])
        delta = x[1:] - x[:-1]
        res['Delta'] = pd.Series(delta, index=indices[1:])
        return res
