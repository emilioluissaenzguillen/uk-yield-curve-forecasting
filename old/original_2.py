# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 17:29:52 2022

@author: ESaenz
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:52:05 2022

@author: addj700
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

import os
wd = os.getcwd()
wd

import warnings
warnings.filterwarnings('ignore')

import math
from math import sqrt
import datetime
from datetime import date, timedelta
from fredapi import Fred
import quandl
import numpy as np
import pandas as pd
from pandas import DataFrame, Series 
import scipy
import scipy.stats as scs
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

from prophet import Prophet # for forecasting via ML called "Prophet"
from prophet.plot import plot_plotly
import logging
logging.getLogger().setLevel(logging.ERROR) # mutes unimportant diagnostic messages


import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.style.use('ggplot')
%matplotlib inline
import matplotlib.pyplot as plt


# standardizes the size of all plots, rather than typing plot(figsize = (12,6)) everytime
from pylab import rcParams
rcParams['figure.figsize'] = 12,6 

import seaborn as sns
from plotly import __version__ 
import cufflinks as cf 
from IPython.display import display, HTML, Image
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objects as go

# set plotly to offline mode for easy reproducibilty
import plotly.offline as py
py.init_notebook_mode()



# Deep learning packages
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping

quandl.ApiConfig.api_key = '-kesKc4A8jkkBwyX3rSX'
fred = Fred(api_key='7e70df3f020852ea824275ce0fea70a0')

# 10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

yieldsp = quandl.get('FRED/T10Y3M', start_date = "1982-01-04", end_date="2020-08-23",collapse = 'daily',
                  date_parser=parser)
yieldsp.columns = ['yieldsp']
yieldsp.tail()

# dataset: numpy array that we will convert into a matrix of dataset
# look_back: number of previous time steps to use as input variables to predict the next time period 
# in this case defaulted to 1.
# default will create a dataset where X is the yield spread at a given time (t) 
# and Y is the yield spread at the next time (t + 1).

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Stacked LSTM for yield spread with memory

# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
dataframe = yieldsp
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
#train_size = int(len(dataset) * 0.9)
#test_size = len(dataset) - train_size
#train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
train, test = dataset[:9051,:], dataset[-651:]

# reshape into X=t and Y=t+1
look_back = 50
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

len(trainX)

len(testX)

len(dataset)

# have an even number of observations in the yieldsp dataset for it to work
# create and fit the LSTM network
# train and test data have 7650 and 1875 samples(or observations), respectively
# => batch size should be a common factor of the train and test size => highest common factor is 75
# dataset will be divided into 7650/75= 102 batches, each with 75 samples
# 1 epoch involves 102 batches or 102 updates 
# if there were 100 epochs => 102*100 = 10,200 updates/batches during the entire training process

# gain finer control of the internal LSTM state by making it "stateful"
# => can build state over the entire training sequence and even maintain that state if needed to make predictions.

batch_size = 100
model = Sequential()
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='RMSprop')

for i in range(100):
    model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()
 
 # make predictions
trainPredict = model.predict(trainX, batch_size=batch_size)

model.reset_states()
testPredict = model.predict(testX, batch_size=batch_size)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.5f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.5f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)- 39 + (look_back*2)+1:len(dataset)-1, :] = testPredict

# combine the predicted and actual data
train_pred = pd.DataFrame(trainPredictPlot, columns = ['train_pred'])
test_pred = pd.DataFrame(testPredictPlot, columns = ['test_pred'])
pred = train_pred.combine_first(test_pred)
pred1 = pred[['train_pred', 'test_pred']]
merged = pd.merge(yieldsp, pred1, on = yieldsp.index).rename(columns= {'key_0': 'Date'}).set_index('Date')

fig_lstm0 = merged.iplot(asFigure=True, kind='scatter', xTitle='Date', yTitle='Yield Spread')
fig_lstm0