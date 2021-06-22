# -*- coding: utf-8 -*-
from covid.CountryDataProcessor import CountryDataProcessor

import numpy as np

def get_country_info(country):
    if country == 'Belarus':
        country_info = {"name":country,
                                "waves":[0,70000],
                                "waves_num":1,
                                "double_regression_max_num":64,
                                "regression_values":np.arange(38, 43)}
    elif country == 'Italy':
        country_info = {"name":country,
                                "waves":[0,240000],
                                "waves_num":0,
                                "double_regression_max_num":1,
                                "regression_values":np.arange(0, 2)}
    else:
        country_info = {"name":"",
                                "waves":[0,0],
                                "waves_num":0,
                                "double_regression_max_num":1,
                                "regression_values":np.arange(0, 2)}
        print("Haven't information about country {}.".format(country))
    return country_info


def get_country_data_processor(country):
    country_info = get_country_info(country)
    cdp = CountryDataProcessor(country=country_info['name'], waves=country_info['waves'])
    cdp.read_data(with_smooth=True)
    return cdp, country_info
