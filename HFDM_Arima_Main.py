import pandas as pd
from datetime import datetime
from util import timeseries_plot, config_plot
from myArima import *

config_plot()

parse_dates = [['Date', 'Time']]
train_file = '../HFDM/广东省/广东-周-发病人数-分地区 - 矫正后 - 副本.xlsx'
test_file = '../HFDM/广东省/gd2019.csv'
train_GD = pd.read_excel(train_file)
test_GD = pd.read_csv(test_file, header=None)
test_GD.columns = train_GD.columns
train_GD['date'] = train_GD['city']
test_GD['date'] = [datetime.strptime(date, '%Y/%m/%d') for date in test_GD['city']]

del train_GD['city']
del test_GD['city']
all_GD = pd.concat((train_GD, test_GD), axis=0)
all_GD.reset_index()
all_GD.index.name = 'date'

# timeseries_plot(x=all_GD['date'], y=all_GD['广州市'], color='g', y_label='广州市')

train_data = train_GD['广州市']
test_data = test_GD['广州市']

# "Grid search" of seasonal ARIMA model.
arima_para = {}
arima_para['p'] = range(4)
arima_para['d'] = range(3)
arima_para['q'] = range(4)
arima_para['P'] = range(4)
arima_para['D'] = range(3)
arima_para['Q'] = range(4)
# the seasonal periodicy is  24 hours
seasonal_para = 52
arima = Arima_Class(arima_para, seasonal_para)

arima.fit(train_data)

# Prediction on observed data starting on pred_start
# observed and prediction starting dates in plots
# One-step ahead forecasts
dynamic = False
arima.pred(train_data, test_data, dynamic, '广州市')

# Dynamic forecasts
dynamic = True
arima.pred(train_data, test_data, dynamic, '广州市')

# Forecasts to unseen future data
# n_steps = 10  # next 100 * 30 min = 50 hours
# arima.forcast(train_data, n_steps, '广州')
