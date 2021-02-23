# -*- coding: utf-8 -*-
import abc


class BaseModel(metaclass=abc.ABCMeta):
    """Базовый класс для моделей распространения COVID-19

    Attributes
    ----------
    coef_ : list
        Коэффициенты модели
    score : float
        Качество модели

    Methods
    -------
    fit(X) :
        Обучение модели
    predict(start, end, init=None):
        Построение прогноза
    """

    def __init__(self):
        "Инициализация объекта"
        self.coef_ = []
        self.score = 0.0

    @abc.abstractmethod
    def fit(self, X):
        """Обучение модели

        Parameters
        ----------
        X : array
            Данные для обучения.
        """
        pass

    @abc.abstractmethod
    def predict(self, start, end, init=None):
        """Построение прогноза

        Parameters
        ----------
        start : строка даты вида ``'YYYY-MM-DD'``
            Начало прогноза
        end : строка даты вида ``'YYYY-MM-DD'``
            Окончание прогноза
        init : array
            Начальные данные
        """
        pass
