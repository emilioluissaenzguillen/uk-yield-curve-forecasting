# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 17:59:16 2022

@author: ESaenz
"""

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

import os
os.chdir('C:\\Users\\MEF08\\Dropbox\\Bayes Business School\\PhD\\Modules\\1st Term\\Financial Econometrics (SMM943)\\Presentation - ML\\Simulation\\input\\glcrealddata')

# UK instantaneous implied real forward curve (gilts)

variable_names_1 = ['30','36','42','48','54','60','66','72','78','84','90','96','102','108','114','120','126','132','138','144','150','156','162','168','174','180','186','192','198','204','210','216','222','228','234','240','246','252','258','264','270','276','282','288','294','300']
variable_names_2 = ['30','36','42','48','54','60','66','72','78','84','90','96','102','108','114','120','126','132','138','144','150','156','162','168','174','180','186','192','198','204','210','216','222','228','234','240','246','252','258','264','270','276','282','288','294','300','306','312','318','324','330','336','342','348','354','360','366','372','378','384','390','396','402','408','414','420','426','432','438','444','450','456','462','468','474','480']


data_1 = pd.read_excel('GLC Real daily data_1985 to 1989.xlsx', sheet_name='2.  real fwd curve', skiprows = 5, parse_dates=True, index_col=0)
data_1.columns = variable_names_1  
data_2 = pd.read_excel('GLC Real daily data_1990 to 1994.xlsx', sheet_name='2.  real fwd curve', skiprows = 5, parse_dates=True, index_col=0)
data_2.columns = variable_names_1  
data_3 = pd.read_excel('GLC Real daily data_1995 to 1999.xlsx', sheet_name='2.  real fwd curve', skiprows = 5, parse_dates=True, index_col=0)
data_3.columns = variable_names_1  
data_4 = pd.read_excel('GLC Real daily data_2000 to 2004.xlsx', sheet_name='2.  real fwd curve', skiprows = 5, parse_dates=True, index_col=0)
data_4.columns = variable_names_1  
data_5 = pd.read_excel('GLC Real daily data_2005 to 2015.xlsx', sheet_name='2. fwd curve',skiprows = 5, parse_dates=True, index_col=0)
data_5.columns = variable_names_1  
data_6 = pd.read_excel('GLC Real daily data_2016 to present.xlsx', sheet_name='2. fwd curve',skiprows = 5, parse_dates=True, index_col=0)
data_6.columns = variable_names_2  
data_6.drop(['306','312','318','324','330','336','342','348','354','360','366','372','378','384','390','396','402','408','414','420','426','432','438','444','450','456','462','468','474','480'], axis=1, inplace=True)

yields = pd.concat([data_1, data_2, data_3, data_4, data_5, data_6])
del [data_1, data_2, data_3, data_4, data_5, data_6]
yields = yields.dropna(how = 'all')
yields.index.name = 'Date'

# Filter data to avoid too many NaNs
yields = yields[~(yields.index < "24-May-2000")]
yields.drop(['30','36','42'], axis=1, inplace=True)

yields.tail()

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

trainPredict_df = pd.DataFrame(columns=yields.columns)
testPredict_df = pd.DataFrame(columns=yields.columns)

trainY_df = pd.DataFrame(columns=yields.columns)
testY_df = pd.DataFrame(columns=yields.columns)

# fix random seed for reproducibility
np.random.seed(7)

for i in yields.columns:
    # load the dataset
    dataframe = pd.DataFrame(yields.loc[:,'48'])
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
 

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
            
    trainPredict_df[i] = pd.DataFrame(scaler.inverse_transform(trainPredict))
    trainY_df[i] = pd.DataFrame(np.transpose(scaler.inverse_transform([trainY])))
    testPredict_df[i] = pd.DataFrame(scaler.inverse_transform(testPredict))
    testY_df[i] = pd.DataFrame(np.transpose(scaler.inverse_transform([testY])))




