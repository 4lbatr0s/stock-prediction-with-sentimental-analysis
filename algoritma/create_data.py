from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


from urllib.request import urlopen
import certifi
import json

from predictions import make_prediction
import yfinance as yf
# ROC = (net income - dividends) / (debt + equity)
x = yf.Ticker("AAPL")
# print(x.financials)
# print(x.dividends)
# print(x.balancesheet)


def get_ev(ticker):
    url = ("https://financialmodelingprep.com/api/v3/enterprise-values/" +
           ticker+"?apikey=5717e43835bcff2c8382bdc2d6f55a3e")
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    result = json.loads(data)
    # print(result)
    return result[0]["enterpriseValue"]


def prediction_last_close_price_comparision(ticker, prediction):
    temp = pd.read_excel("DB_{}.xlsx".format(ticker))
    temp = temp["Close"].values
    temp = np.array(temp)
    last_close_price = temp[-1]
    result = abs(prediction-last_close_price) / prediction
    return result


ticker_list = ['TSLA', 'MSFT']
market_cap_value_list = {}
ebit_list = {}  # ebit value
ev_list = {}  # entreprise value
ebit_ev_list = {}  # ebit/entreprise value
roc_list = {}  # return on capital : sermaye getirisi
predictions = {}
stock_last_value_prediction_value_difference = {}
day = 5
for i in ticker_list:
    # print(i)
    ticker = yf.Ticker(i)
    financial_values = ticker.financials
    ebit = financial_values.loc["Ebit"][0]
    ebit_list[i] = ebit
    ev = get_ev(i)
    ev_list[i] = ev
    ebit_ev_list[i] = (ebit_list[i]/ev_list[i])
    # ROC = (net income - dividends) / (debt + equity) #API YARDIMIYLA ÇEKİLECEK.
    roc_list[i] = 5000000
    predictions[i] = make_prediction(
        '{}.h5'.format(i), day, 'DB_{}.xlsx'.format(i))
    stock_last_value_prediction_value_difference[i] = prediction_last_close_price_comparision(
        i, predictions[i])
