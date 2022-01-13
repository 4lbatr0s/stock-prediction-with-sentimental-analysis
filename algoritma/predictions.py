from sys import modules
from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def make_prediction(model, day, file):
    new_model = load_model('./models/{}'.format(model))
    count_day = day
    merged_data = pd.read_excel(file)
    keep_columns = ['Open', 'High', 'Low', 'Volume',
                    'Subjectivity', 'Polarity', 'Compound']
    all_input_values = merged_data[keep_columns].values  # numpy array.
    all_close_values = merged_data['Close'].values
    model_close_values = []

    for i in range(count_day, len(all_close_values)):
        model_close_values.append(all_close_values[i])
        training_size = int(len(model_close_values)*0.75)

    all_close_values_train, all_close_values_test = model_close_values[0:training_size], model_close_values[training_size:len(
        model_close_values)]
    all_close_values_train = np.array(all_close_values_train)
    all_close_values_test = np.array(all_close_values_test)
    all_close_values_train = np.array([all_close_values_train]).T
    all_close_values_test = np.array([all_close_values_test]).T
    scaler = StandardScaler()
    all_close_values_train = scaler.fit_transform(all_close_values_train)
    all_close_values_test = scaler.transform(all_close_values_test)
    last_count_days = all_input_values[len(all_input_values)-count_day:]
    last_count_days = np.array(last_count_days)
    last_count_days = np.reshape(
        last_count_days, (1, (last_count_days.shape[0]*last_count_days.shape[1])))
    prediction = new_model.predict(last_count_days)
    prediction = scaler.inverse_transform(prediction)
    prediction = prediction.flatten()
    return prediction


# print(make_prediction('MSFT.h5', 5, 'DB_MSFT.xlsx'))
# print(make_prediction("TSLA.h5", 5, "DB_TSLA.xlsx"))
