# -*- coding: utf-8 -*-
"""
SMM943 Financial Econometrics - Coursework      
Author: Emilio Luis Saenz Guillen, PhD Student in Actuarial Science  
Professor: Giovanni Urga                                         
Bayes Business School                                               
December 2022 
     
"""
from pathlib import Path

try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        ipython.magic('reset -sf')
except Exception:
    pass

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import logging
logging.getLogger().setLevel(logging.ERROR) # mutes unimportant diagnostic messages


# Deep learning packages
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping


import os

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
INPUT_DIR = PROJECT_ROOT / "input" / "glcrealddata"
OUTPUT_DIR = PROJECT_ROOT / "output"

os.chdir(INPUT_DIR)

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

yields.to_csv(OUTPUT_DIR / "yields.csv")


# Loop
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

 
# 95 percent of total obs in train set, 5 percent in test set
    train_size = int(len(dataset) * 0.95)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    
# look_back: number of previous time steps to use as input variables to predict
# the next time period (defaulted to 1)

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    
    look_back = 30
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)


# reshape input into 3D as [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], look_back, 1)
    X_test = X_test.reshape(X_test.shape[0], look_back, 1)
    
    import random
    random.seed(190)
    model = Sequential()

# use 100 neurons in the first hidden layer and 1 neuron in the output layer to predict yields
# input shape is look_back time steps with one feature
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


mae_mse = pd.DataFrame(columns=yields.columns, index=["mae_train","mse_train","mae_test","mse_test"])


for i in yields.columns:
    mae_mse.iloc[0][i] = mean_absolute_error(Y_train_df[i].to_numpy(), train_predict_df[i].to_numpy())
    mae_mse.iloc[1][i] = np.sqrt(mean_squared_error(Y_train_df[i].to_numpy(), train_predict_df[i].to_numpy()))
    mae_mse.iloc[2][i] = mean_absolute_error(Y_test_df[i].to_numpy(), test_predict_df[i].to_numpy())
    mae_mse.iloc[3][i] = np.sqrt(mean_squared_error(Y_test_df[i].to_numpy(), test_predict_df[i].to_numpy()))

mae_mse.to_csv(OUTPUT_DIR / "mae_mse.csv")

# Get a panel with the actual and modeled series

merged = {}
model = {}

for i in yields.columns:
    # shift train predictions
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = pd.DataFrame(train_predict_df[i])

    # shift test predictions
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2):len(dataset), :] = pd.DataFrame(test_predict_df[i])

    # combine the predicted and actual data
    train_pred = pd.DataFrame(trainPredictPlot, columns = ['train_pred'])
    test_pred = pd.DataFrame(testPredictPlot, columns = ['test_pred'])
    pred = train_pred.combine_first(test_pred)
    pred1 = pred[['train_pred', 'test_pred']]   
    model[i] = pred1['train_pred'].combine_first(pred1['test_pred']) # this df I use it for calculating the equiv rate
    merged[i] = pd.merge(yields[i], pred1, on = yields.index).rename(columns= {'key_0': 'Date'}).set_index('Date')
    merged.get(i)['actual']=merged.get(i).pop(i)

model = pd.DataFrame.from_dict(model)
model = model.set_index(yields.index)

merged_df = pd.DataFrame(columns=yields.columns,index=yields.index).add_suffix('_actual')
merged_df = pd.concat([merged_df,pd.DataFrame(columns=yields.columns).add_suffix('_train')])
merged_df = pd.concat([merged_df,pd.DataFrame(columns=yields.columns).add_suffix('_test')])
merged_df = merged_df.loc[:,['48_actual','48_train','48_test',	'54_actual','54_train','54_test',	'60_actual','60_train','60_test',	'66_actual','66_train','66_test',	'72_actual','72_train','72_test',	'78_actual','78_train','78_test',	'84_actual','84_train','84_test',	'90_actual','90_train','90_test',	'96_actual','96_train','96_test',	'102_actual','102_train','102_test',	'108_actual','108_train','108_test',	'114_actual','114_train','114_test',	'120_actual','120_train','120_test',	'126_actual','126_train','126_test',	'132_actual','132_train','132_test',	'138_actual','138_train','138_test',	'144_actual','144_train','144_test',	'150_actual','150_train','150_test',	'156_actual','156_train','156_test',	'162_actual','162_train','162_test',	'168_actual','168_train','168_test',	'174_actual','174_train','174_test',	'180_actual','180_train','180_test',	'186_actual','186_train','186_test',	'192_actual','192_train','192_test',	'198_actual','198_train','198_test',	'204_actual','204_train','204_test',	'210_actual','210_train','210_test',	'216_actual','216_train','216_test',	'222_actual','222_train','222_test',	'228_actual','228_train','228_test',	'234_actual','234_train','234_test',	'240_actual','240_train','240_test',	'246_actual','246_train','246_test',	'252_actual','252_train','252_test',	'258_actual','258_train','258_test',	'264_actual','264_train','264_test',	'270_actual','270_train','270_test',	'276_actual','276_train','276_test',	'282_actual','282_train','282_test',	'288_actual','288_train','288_test',	'294_actual','294_train','294_test',	'300_actual','300_train','300_test']]
col=0
for i in yields.columns:    
    merged_df.iloc[:,col] = pd.DataFrame.from_dict(merged.get(i).actual)
    merged_df.iloc[:,col+1] = pd.DataFrame.from_dict(merged.get(i).train_pred)
    merged_df.iloc[:,col+2] = pd.DataFrame.from_dict(merged.get(i).test_pred)
    col = col+3

merged_df.to_csv(OUTPUT_DIR / "merged.csv")



    
# Equivalent interest rate
from scipy.optimize import fsolve

maturities = [4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25]

# Yields
discount_rates_sum =pd.DataFrame(columns=["discount_rates_sum"], index=yields.index)
equiv_rate =pd.DataFrame(columns=["equiv_rate"], index=yields.index)

for i in range(0, len(yields)):
    t = 0
    for j in range(0, len(maturities)):
        s = 1/((1+(yields.iloc[i,j]/100))**maturities[j])     
        t = t+s         
    discount_rates_sum.iloc[i] = t    

# Define the expression whose roots we want to findmport numpy as np
for i in range(0, len(yields)):    
    def myFunction(equiv_rate):
        t = 0
        for j in range(0, len(maturities)):
            s = 1/((1+(equiv_rate/100))**maturities[j])
            t = t+s
        F = t - discount_rates_sum.iloc[i]
        return F
    initial_guess= 0.5
    equiv_rate.iloc[i] = fsolve(myFunction,0.5)

# Model (LSTM)
model = model.dropna()
discount_rates_sum_model =pd.DataFrame(columns=["discount_rates_sum_model"], index=model.index)
equiv_rate_model =pd.DataFrame(columns=["equiv_rate_model"], index=model.index)

for i in range(0, len(model)):
    t = 0
    for j in range(0, len(maturities)):
        s = 1/((1+(model.iloc[i,j]/100))**maturities[j])     
        t = t+s         
    discount_rates_sum_model.iloc[i] = t    

# Define the expression whose roots we want to findmport numpy as np
for i in range(0, len(model)):    
    def myFunction(equiv_rate):
        t = 0
        for j in range(0, len(maturities)):
            s = 1/((1+(equiv_rate/100))**maturities[j])
            t = t+s
        F = t - discount_rates_sum_model.iloc[i]
        return F
    initial_guess= 0.5
    equiv_rate_model.iloc[i] = fsolve(myFunction,0.5)

equiv_rate_merged = pd.concat([equiv_rate, equiv_rate_model], axis=1)
equiv_rate_merged.to_csv(OUTPUT_DIR / "equiv_rate_merged.csv")

equiv_rate_merged = equiv_rate_merged.dropna()
equiv_rate_train = equiv_rate_merged[~(equiv_rate_merged.index >= "25-Nov-2021")]
equiv_rate_test = equiv_rate_merged[~(equiv_rate_merged.index < "25-Nov-2021")]

print('Train equiv rate Mean Absolute Error:', mean_absolute_error(equiv_rate_train.equiv_rate.to_numpy(), equiv_rate_train.equiv_rate_model.to_numpy()))
print('Train equiv rate Root Mean Squared Error:',np.sqrt(mean_squared_error(equiv_rate_train.equiv_rate.to_numpy(), equiv_rate_train.equiv_rate_model.to_numpy())))
print('Test equiv rate Mean Absolute Error:', mean_absolute_error(equiv_rate_test.equiv_rate.to_numpy(), equiv_rate_test.equiv_rate_model.to_numpy()))
print('Test equiv rate Root Mean Squared Error:',np.sqrt(mean_squared_error(equiv_rate_test.equiv_rate.to_numpy(), equiv_rate_test.equiv_rate_model.to_numpy())))

# Model (Diebold-Li)
model_dbli = pd.read_csv(OUTPUT_DIR / "MMSEForecasts.csv", header=None)
model_dbli.columns=yields.columns
ind = pd.DataFrame(yields.index)
ind.drop(ind.loc[ind['Date'] < '2021-11-25 00:00:00'].index, inplace=True)
ind= ind.set_index('Date')
model_dbli.index = ind.index


discount_rates_sum_model_dbli =pd.DataFrame(columns=["discount_rates_sum_model_dbli"], index=model_dbli.index)
equiv_rate_model_dbli =pd.DataFrame(columns=["equiv_rate_model_dbli"], index=model_dbli.index)

for i in range(0, len(model_dbli)):
    t = 0
    for j in range(0, len(maturities)):
        s = 1/((1+(model_dbli.iloc[i,j]/100))**maturities[j])     
        t = t+s         
    discount_rates_sum_model_dbli.iloc[i] = t    

# Define the expression whose roots we want to find
for i in range(0, len(model_dbli)):    
    def myFunction(equiv_rate):
        t = 0
        for j in range(0, len(maturities)):
            s = 1/((1+(equiv_rate/100))**maturities[j])
            t = t+s
        F = t - discount_rates_sum_model_dbli.iloc[i]
        return F
    initial_guess= 0.5
    equiv_rate_model_dbli.iloc[i] = fsolve(myFunction,0.5)

equiv_rate_merged_dbli = pd.concat([equiv_rate, equiv_rate_model_dbli], axis=1)
equiv_rate_merged_dbli = equiv_rate_merged_dbli.dropna()
equiv_rate_merged_dbli.to_csv(OUTPUT_DIR / "equiv_rate_merged_dbli.csv")


print('Test equiv rate Mean Absolute Error:', 
      mean_absolute_error(equiv_rate_merged_dbli.equiv_rate.to_numpy(),
                          equiv_rate_merged_dbli.equiv_rate_model_dbli.to_numpy()))
print('Test equiv rate Root Mean Squared Error:',np.sqrt(mean_squared_error(equiv_rate_merged_dbli.equiv_rate.to_numpy(), equiv_rate_merged_dbli.equiv_rate_model_dbli.to_numpy())))

i = '300'
print('Test Mean Absolute Error:',
      mean_absolute_error(Y_test_df[i].to_numpy(),
                          model_dbli[i].to_numpy()))
print('Test Root Mean Squared Error:',
      np.sqrt(mean_squared_error(Y_test_df[i].to_numpy(),
                                 model_dbli[i].to_numpy())))

# Diebold-Mariano test
from dm_test import dm_test

equiv_rate_merged= pd.concat([equiv_rate, equiv_rate_model, equiv_rate_model_dbli], axis=1)
equiv_rate_merged = equiv_rate_merged.dropna()


rt = dm_test(equiv_rate_merged.equiv_rate,
             equiv_rate_merged.equiv_rate_model_dbli,
             equiv_rate_merged.equiv_rate_model,h = 1, crit="MAD")

print(rt)

rt = dm_test(equiv_rate_merged.equiv_rate,
             equiv_rate_merged.equiv_rate_model_dbli,
             equiv_rate_merged.equiv_rate_model,h = 1, crit="MSE")

print(rt)
