import pandas as pd
import numpy as np
from tensorflow import keras as kr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import Dense, Dropout, Flatten
from keras.models import load_model
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt



def create_value_sets(Ticker,count):
    count_day = count
    merged_data = pd.read_excel("DB_{0}.xlsx".format(Ticker))
    keep_columns=['Open','High','Low','Volume','Subjectivity','Polarity','Compound']
    all_input_values=merged_data[keep_columns].values #numpy array.
    model_input = create_model_input(all_input_values, count_day)
    model_input=np.array(model_input)
    all_close_values = merged_data['Close'].values
    model_close_values=[]
    for i in range(count_day,len(all_close_values)):
        model_close_values.append(all_close_values[i])
    model_close_values=np.array(model_close_values)
    return model_input, model_close_values


def create_model_input(all_input_values, count):
    model_input=[]
    count_day=count

    temp=[]
    for i in range(0,len(all_input_values)-count_day):
        temp=all_input_values[i:i+count_day]    
        model_input.append(temp)
    return model_input


def create_train_test_data_sets():
    training_size=int(len(model_input)*0.75)
    model_input,model_close_values = create_value_sets(Tic)
    test_size=len(model_input)-training_size
    all_input_values_train, all_input_values_test=model_input[0:training_size,:],model_input[training_size:len(model_input),:]

    training_size=int(len(model_close_values)*0.75)
    all_close_values_train, all_close_values_test = model_close_values[0:training_size], model_close_values[training_size:len(model_close_values)]