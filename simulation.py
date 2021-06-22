# -*- coding: utf-8 -*-

from covid.CountryDataProcessor import CountryDataProcessor as CDP
from covid.cdp_fabric import get_country_data_processor
from covid.plotting import plot_weights, plot_infected
from covid.preprocess import smooth_average
from covid.utils import read, calc_deltas

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy as pt
import seaborn as sns
import sklearn.linear_model as lm

country = 'Belarus'
num_wave = 1
N = 40

cdp, country_info = get_country_data_processor(country)

data_cut = cdp.get_wave(num_wave)
deltas, features, tex_features = cdp.calc_deltas(N, data_cut.index)

l = 10
coef_ = np.ones(len(features))
h = 1/(N+1-l)
coef_[l:] = np.array([i*h for i in range(1, N+1-l)][::-1])
y_lin = deltas[features].dot(coef_)
exp_coef_  = np.ones(len(features))
exp_coef_[-1] = 1e-1
k = np.log(exp_coef_[-1])/(l-N)
idx = [k*(l-i-1) for i in range(l, N-1)]
exp_coef_[l:-1] = np.exp(idx)
y_exp = deltas[features].dot(exp_coef_)
