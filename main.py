from typing import final
import pandas as pd
import numpy as np
from tensorflow import keras as kr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from keras.models import load_model
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
# import final_model
import yfinance as yf
from yfinance import ticker
from urllib.request import urlopen
import certifi
import json

# last_days_for_msft = pd.read_excel("last_days_msft.xlsx")
# last_days_for_msft = np.array(last_days_for_msft)
# print(last_days_for_msft.shape)
# last_days_for_msft = last_days_for_msft.reshape(-1,1)
# result_msft=final_model.prediction2
# It can be used to reconstruct the model identically.


# def create_value_sets(Ticker,count):
#     count_day = count
#     merged_data = pd.read_excel("DB_{0}.xlsx".format(Ticker))
#     keep_columns=['Open','High','Low','Volume','Subjectivity','Polarity','Compound']
#     all_input_values=merged_data[keep_columns].values #numpy array.
#     model_input = create_model_input(all_input_values, count_day)
#     model_input=np.array(model_input)
#     all_close_values = merged_data['Close'].values
#     model_close_values=[]
#     for i in range(count_day,len(all_close_values)):
#         model_close_values.append(all_close_values[i])
#     model_close_values=np.array(model_close_values)
#     return model_input, model_close_values


# def create_model_input(all_input_values, count):
#     model_input=[]
#     count_day=count

#     temp=[]
#     for i in range(0,len(all_input_values)-count_day):
#         temp=all_input_values[i:i+count_day]
#         model_input.append(temp)
#     return model_input


# def create_train_test_data_sets():
#     training_size=int(len(model_input)*0.75)
#     model_input,model_close_values = create_value_sets(Tic)
#     test_size=len(model_input)-training_size
#     all_input_values_train, all_input_values_test=model_input[0:training_size,:],model_input[training_size:len(model_input),:]

#     training_size=int(len(model_close_values)*0.75)
#     all_close_values_train, all_close_values_test = model_close_values[0:training_size], model_close_values[training_size:len(model_close_values)]

# model = load_model("msft.h5")
# prediction = model.predict(final_model.last_count_days)

# print(final_model.prediction2)

# predicitonların diğer dosyalardan nasıl okunacağı sorulacak!
# url = ("https://financialmodelingprep.com/api/v3/financial-statement-symbol-lists?apikey=5717e43835bcff2c8382bdc2d6f55a3e")
# url =("https://financialmodelingprep.com/api/v3/enterprise-values/"+ticker)

def get_ev(ticker):
    url = ("https://financialmodelingprep.com/api/v3/enterprise-values/" +
           ticker+"?apikey=5717e43835bcff2c8382bdc2d6f55a3e")
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    result = json.loads(data)
    print(result)
    return result[0]["enterpriseValue"]


prediction_msft = 351.61
prediction_tsla = 969.50


ticker = yf.Ticker("TSLA")
print(ticker.financials)
print(ticker.cashflow)
print(ticker.earnings)
print(ticker.actions)
print(ticker.balance_sheet)
print(ticker.balancesheet)

# ticker_list = ['TSLA', 'MSFT']
# ebit_list = []
# ev_list = []
# roc = []  # return on capital : sermaye getirisi
# for i in ticker_list:
#     print(i)
#     ticker = yf.Ticker(i)
#     financial_values = ticker.financials
#     ebit = financial_values.loc["Ebit"][0]
#     ebit_list.append(ebit)
#     ev = get_ev(i)
#     ev_list.append(ev)
#     # roc_ = ticker.fi


# data = {'Stock ': ['TSLA', 'MSTF']}

# data["Ebit"] = ebit_list
# data["EV"] = ev_list
# print(data)
