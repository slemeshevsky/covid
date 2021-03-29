# -*- coding: utf-8 -*-

"""
Модуль содержит классы для моделей основанных на уравнении с запаздываением

.. math::

   \dot{x}(t) = r (x(t)-x(t-l))(K - x(t-1))

"""


from .base import BaseModel
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


class DelayEquation(BaseModel):
    """Базовый класс для моделей, основанных на аппроксимации уравнения с
запаздыванием

Parameters
----------
periods : Диапазон параметров :math:`l`
    По-умолчанию ``range(3,15)``

    """

    def __init__(self, periods=range(3, 15)):
        """
    Инициализирует объект
        """
        super().__init__()
        self.r = 0.0
        self.K = 0.0
        self.incubation = 0
        self.periods = periods

    def __str__(self):
        s = """Коэффициенты: r={}, K={}
Качество: {}
Время заражения: {}""".format(self.r, self.K, self.score,
                              self.incubation)
        return s

    def __repr__(self):
        s = """Коэффициенты: r={}, K={}
Качество: {}
Время заражения: {}""".format(self.r, self.K, self.score,
                              self.incubation)
        return s


class ExplicitDelayEquation(DelayEquation):
    """
Класс, реализующий модель, основанную на уравнении с запаздыванием

.. math:: x_k - x_{k-1} = r(x_{k-1} - x_{k-l})(K - x_{k-1})

Parameters
----------
periods : Диапазон параметров :math:`l`
    По-умолчанию ``range(3,15)``
    """

    def fit(self, X):
        """
    Строит модель (определяет коэффициенты :math:`r` и :math:`K`) для
    уравнения с запаздыванием.

    Parameters
    ----------
    X : array
        Данные для обучения
    y : array
        Целевые данные

    Returns
    -------
    self : Возвращает экземпляр объекта
        """

        self.score = 0.0

        for p in self.periods:
            s = np.zeros_like(X)
            s[p:] = (X[p:] - X[p-1:-1])/(X[p-1:-1] - X[:-p])
            model = LinearRegression().fit(X[p:, None], s[p:])
            score = model.score(X[p:, None], s[p:])
            if score > self.score:
                self.incubation = p
                self.score = score
                self.coef_ = [model.coef_[0], model.intercept_]
                self.r = -model.coef_[0]
                self.K = -model.intercept_/model.coef_[0]
                self.s = s

        return self

    def _decision_func(self, init, num_periods):
        """
    Функция для прогноза по явному уравнению с запаздыванием

    .. math:: x_k - x_{k-1} = r(x_{k} - x_{k-l})(K - x_{k-1})

    Parameters
    ----------
    init : array размерности incubation
        Начальные данные
    num_periods : int
        Количество дней прогноза

    Returns
    -------

    x : array размерности num_periods
        Содержит значение прогноза
        """
        x = np.zeros(num_periods)
        l = len(init)

        if l != self.incubation:
            raise ValueError("Длина массива начальных данных должна быть {}".format(self.incubation))

        x[:l] = init[:]
        for k in range(l, len(x)):
            s = self.coef_[1] + self.coef_[0]*x[k-1]
            x[k] = x[k-1] + s*(x[k-1]-x[k-l])

        return x

    def predict(self, init, start, end):
        """
    Построение прогноза

    Parameters
    ----------

    init : array размерности incubation
    start : строка ``'YYYY-MM-DD'`` даты начала прогноза
    end : строка ``'YYYY-MM-DD'`` даты окончания прогноза

    Returns
    -------
    predict : pandas.DataFrame
        Содержит значения прогноза
        """
        t = pd.date_range(start, end)
        x_values = self._decision_func(init, len(t))
        delta_values = x_values[1:] - x_values[:-1]
        predict = pd.DataFrame(x_values, index=t, columns=['X'])
        predict['dX'] = pd.Series(delta_values, index=t[1:])

        return predict


class ImplicitDelayEquation(DelayEquation):
    """
Класс, реализующий модель, основанную на неявном уравнении с запаздыванием

.. math:: x_k - x_{k-1} = r(x_{k} - x_{k-l})(K - x_{k-1})

Parameters
----------
periods : Диапазон параметров l
    По-умолчанию range(3,15)
    """

    def fit(self, X):
        """
    Строит модель (опоредлеят коэффициенты :math:`r` и :math:`K`) для
    уравнения с запаздыванием.

    Parameters
    ----------
    X : array
        Данные для обучения
    y : array
        Целевые данные

    Returns
    -------
    self : Возвращает экземпляр объекта
        """

        self.score = 0.0
        self.scores = dict()

        for p in self.periods:
            s = np.zeros_like(X)
            s[p:] = (X[p:] - X[p-1:-1])/(X[p:] - X[:-p])
            model = LinearRegression().fit(X[p:, None], s[p:])
            score = model.score(X[p:, None], s[p:])
            self.scores[p] = score
            if score > self.score:
                self.incubation = p
                self.score = score
                self.coef_ = [model.coef_[0], model.intercept_]
                self.r = -model.coef_[0]
                self.K = -model.intercept_/model.coef_[0]
                self.s = s

        return self

    def _decision_func(self, init, num_days):
        """
    Функция для прогноза по явному уравнению с запаздыванием

    .. math:: x_k - x_{k-1} = r(x_{k} - x_{k-l})(K - x_{k-1})

    Parameters
    ----------
    init : array размерности incubation
        Начальные данные
    num_periods : int
        Количество дней прогноза

    Returns
    -------
    x : array размерности num_periods
        Содержит значение прогноза
        """
        x = np.zeros(num_days)
        l = len(init)

        if l != self.incubation:
            raise ValueError("Длина массива начальных данных должна быть {}".format(self.incubation))

        x[:l] = init[:]
        for k in range(l, len(x)):
            s = self.coef_[1] + self.coef_[0]*x[k-1]
            x[k] = (x[k-1] - s*x[k-l])/(1-s)

        return x

    def predict(self, init, start, end):
        """
    Построение прогноза

    Parameters
    ----------
    init : array размерности incubation
    start : строка ``'YYYY-MM-DD'`` даты начала прогноза
    end : строка ``'YYYY-MM-DD'`` даты окончания прогноза

    Returns
    -------
    predict : pandas.DataFrame
        Содержит значения прогноза
        """

        t = pd.date_range(start, end)
        x_values = self._decision_func(init, len(t))
        delta_values = x_values[1:] - x_values[:-1]
        predict = pd.DataFrame(x_values, index=t, columns=['X'])
        predict['dX'] = pd.Series(delta_values, index=t[1:])

        return predict
