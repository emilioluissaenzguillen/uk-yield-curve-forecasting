# -*- coding: utf-8 -*-
"""
SMM943 Financial Econometrics - Coursework      
Author: Emilio Luis Saenz Guillen, PhD Student in Actuarial Science  
Professor: Giovanni Urga                                         
Bayes Business School                                               
December 2022 
     
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

# Market Yield on U.S. Treasury Securities at 30-Year Constant Maturity
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

treas = ['FRED/DGS3MO',
           'FRED/DGS6MO',
           'FRED/DGS1',
           'FRED/DGS2',
           'FRED/DGS3',
           'FRED/DGS5',
           'FRED/DGS7',
           'FRED/DGS10',
           'FRED/DGS20',
           'FRED/DGS30']

yields = quandl.get(treas, start_date = "1993-10-01", end_date="2022-07-15",collapse = 'daily',
                  date_parser=parser)
yields.columns = ['tres3mo', 
                    'tres6mo',
                    'tres1y',
                    'tres2y',
                    'tres3y',
                    'tres5y',
                    'tres7y',
                    'tres10y',
                    'tres20y',
                    'tres30y']

yields.tail()

# Graph
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.io as pio
pio.renderers.default = 'browser'

x = yields.columns
y = yields.index
z = yields.to_numpy()
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
fig.update_layout(title='Yield Curves',
                  scene = {"aspectratio": {"x": 1, "y": 1, "z": 0.4}})
fig.show()

# Train-test split : last 10 percent of the observations are in the test set"

train_predict_df = pd.DataFrame(columns=yields.columns)
test_predict_df = pd.DataFrame(columns=yields.columns)

Y_train_df = pd.DataFrame(columns=yields.columns)
Y_test_df = pd.DataFrame(columns=yields.columns)


for i in yields.columns:
    dataset = yields[i].values #numpy.ndarray
    dataset = dataset.astype('float32') # ensures that all the data is float and occupies less space
    dataset = np.reshape(dataset, (-1, 1)) 


    scaler = MinMaxScaler(feature_range=(0, 1)) # rescale the data => normalize to lie between 0 and 1. 
    dataset = scaler.fit_transform(dataset)

# calculates the index of the split point and separates the data into the training datasets 
# model created on the 80 percent of total obs in train set, 20 percent is in the test set

    train_size = int(len(dataset) * 0.95)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    test_size

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    
    look_back = 30
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)


# reshape input into 3D as [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    yields.shape

    train, test = np.split(yields, [int(.95 *len(yields))])
    train.shape

    import random
    random.seed(190)
    model = Sequential()

# use 100 neurons in the first hidden layer and 1 neuron in the output layer to predict yieldsp
# input shape is one time step with features
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))

# regularize: dropout 20 percent of the data
    model.add(Dropout(0.2))
    model.add(Dense(1))

# Use the MSE loss function and the efficient Adam version of stochastic gradient descent.
# Loss from RMSprop < loss from Adadelta (slowest) < loss from Adam (fastest)
# train on len(X_train) and validates on len(X_test) number of features

    model.compile(loss='mean_squared_error', optimizer='RMSprop')

# train the network for 1000 epochs 
    history = model.fit(X_train, Y_train, epochs= 1000, batch_size=100, validation_data=(X_test, Y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

# Training Phase
    model.summary()

# make predictions    
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

# invert predictions before calculating the error scores
# => ensures that the performance is reported in the same units as the original data

    train_predict_df[i] = pd.DataFrame(scaler.inverse_transform(train_predict))
    Y_train_df[i] = pd.DataFrame(np.transpose(scaler.inverse_transform([Y_train])))
    test_predict_df[i] = pd.DataFrame(scaler.inverse_transform(test_predict))
    Y_test_df[i] = pd.DataFrame(np.transpose(scaler.inverse_transform([Y_test])))

# estimate the model's performance on the train and test set 
# gives point of comparison for the new models
for i in yields.columns:
    print('Train Mean Absolute Error:', mean_absolute_error(Y_train_df[i].to_numpy(), train_predict_df[i].to_numpy()))
    print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train_df[i].to_numpy(), train_predict_df[i].to_numpy())))
    print('Test Mean Absolute Error:', mean_absolute_error(Y_test_df[i].to_numpy(), test_predict_df[i].to_numpy()))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test_df[i].to_numpy(), test_predict_df[i].to_numpy())))


mae_mse = pd.DataFrame(columns=yields.columns, index=["mae","mse"])

















