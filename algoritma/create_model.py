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


def create_model(Ticker, day):
    merged_data = pd.read_excel("DB_{}.xlsx".format(Ticker))
    keep_columns = ['Open', 'High', 'Low', 'Volume',
                    'Subjectivity', 'Polarity', 'Compound']
    all_input_values = merged_data[keep_columns].values  # numpy array.

    model_input = []
    count_day = day

    temp = []
    for i in range(0, len(all_input_values)-count_day):
        temp = all_input_values[i:i+count_day]
        model_input.append(temp)

    print("len(model_input): ", len(model_input))

    model_input = np.array(model_input)
    print("model_input.shape: ", model_input.shape)

    all_close_values = merged_data['Close'].values

    model_close_values = []
    for i in range(count_day, len(all_close_values)):
        model_close_values.append(all_close_values[i])

    model_close_values = np.array(model_close_values)
    print("model_close_values.shape: ", model_close_values.shape)

    training_size = int(len(model_input)*0.75)
    test_size = len(model_input)-training_size
    all_input_values_train, all_input_values_test = model_input[0:training_size, :], model_input[training_size:len(
        model_input), :]

    training_size = int(len(model_close_values)*0.75)
    all_close_values_train, all_close_values_test = model_close_values[0:training_size], model_close_values[training_size:len(
        model_close_values)]

    all_close_values_train = np.array(all_close_values_train)
    all_close_values_test = np.array(all_close_values_test)

    all_input_values_train = np.reshape(all_input_values_train, (
        all_input_values_train.shape[0], all_input_values_train.shape[1]*all_input_values_train.shape[2]))
    print("all_input_values_train.shape: ", all_input_values_train.shape)

    all_input_values_test = np.reshape(all_input_values_test, (
        all_input_values_test.shape[0], all_input_values_test.shape[1]*all_input_values_test.shape[2]))
    print("all_input_values_test.shape: ", all_input_values_test)

    # scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_input_values_train = scaler.fit_transform(
        all_input_values_train)  # hem fit ediyor hem de scale ediyor
    # sadece transform olacak. cünkü model bu veriyi eğitim esnasında görmeyecek
    all_input_values_test = scaler.transform(all_input_values_test)

    all_close_values_train = np.array([all_close_values_train]).T
    all_close_values_test = np.array([all_close_values_test]).T

    scaler2 = StandardScaler()
    all_close_values_train = scaler2.fit_transform(all_close_values_train)
    all_close_values_test = scaler2.transform(all_close_values_test)

    # Taban model ile ana model birleştirilir.

    model = Sequential()
    model.add(LSTM(units=50, input_shape=(
        all_input_values_train.shape[1], 1), return_sequences=True))
    model.add(LSTM(units=50))
    # model.add(LSTM(units=50,return_sequences=True))
    # model.add(LSTM(units=50))
    model.add(Flatten())
    model.add(Dense(units=128))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    opt = kr.optimizers.Adam(lr=0.001, decay=1e-4)
    # learning rate atanabilir.
    model.compile(optimizer=opt, loss='mean_squared_error')

    model.fit(all_input_values_train, all_close_values_train,
              epochs=25, batch_size=16, verbose=1)

    # modeli yakalıyor mu diye bir assertion oluşturmam gerekiyor. eğer modeli yakalamıyorsa bir save oluştursun
    # eğer yakalıyorsa önceki modeli load etsin
    # prediction'un performanslarını her bir iterasyonda data frame yazılabilir.
    # standart sapması alınıp, diğer modellerle karşılaştırılabilir.
    # Grid search
    model.save('./models/{}.h5'.format(Ticker))
    train_predict = model.predict(all_input_values_train)
    test_predict = model.predict(all_input_values_test)

    print('all_close_values_train shape: ', all_close_values_train.shape)

    print('all_close_values_test shape: ', all_close_values_test.shape)
    print('all_close_values_test_pred shape: ', test_predict.shape)
    print('all_close_values_test_pred shape: ', test_predict.shape)

    score = model.evaluate(all_input_values_test, all_close_values_test)
    print("score :", score)

    model.evaluate(all_input_values_train, all_close_values_train)

    test_predict = scaler2.inverse_transform(test_predict)
    train_predict = scaler2.inverse_transform(train_predict)

    all_close_values_test = np.array(all_close_values_test)

    all_close_values_train = scaler2.inverse_transform(all_close_values_train)
    all_close_values_test = scaler2.inverse_transform(all_close_values_test)

    mape_test = mean_absolute_percentage_error(
        all_close_values_test, test_predict)
    mape_train = mean_absolute_percentage_error(
        all_close_values_train, train_predict)
    print("mape_test: ", mape_test)
    print("mape_train: ", mape_train)

    # neden yüksek çıkıyor, mape ile karşılastır tekrar bak
    rmse = np.sqrt(mean_squared_error(all_close_values_test, test_predict))
    print("rmse: ", rmse)

    train_predict = np.array(train_predict)
    train_predict.shape

    last_count_days = all_input_values[len(all_input_values)-count_day:]
    last_count_days = np.array(last_count_days)
    last_count_days = np.reshape(
        last_count_days, (1, (last_count_days.shape[0]*last_count_days.shape[1])))
    prediction = model.predict(last_count_days)
    prediction2 = scaler2.inverse_transform(prediction)

    print("prediction :", prediction2)

    plt.figure(figsize=(16, 8))
    plt.title(
        'Stock Price Prediction {} with Machine Learning LSTM Model'.format(Ticker))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('{} Close Price'.format(Ticker), fontsize=18)
    plt.plot(all_close_values)
    plt.plot(train_predict)
    plt.plot(np.arange(len(test_predict)) +
             len(all_close_values_train), test_predict)
    plt.plot(len(all_input_values)+count_day, prediction2, marker="o",
             markersize=10, markeredgecolor="red", markerfacecolor="green")
    plt.annotate("Prediction", (len(all_input_values) +
                                count_day+10, prediction2))
    plt.legend(['Close Price', 'Training Prediction',
                'Test Prediction'], loc='lower right')
    plt.show()


# create_model("TSLA", 5)

create_model("MSFT", 5)
