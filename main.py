from datetime import date
from keras.models import load_model
from pandas_datareader import data
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os.path


def convertTimeToString(data_list):
    """
    Converts the list of datatime to numpy array of string
    :param data_list: list of datetime objects
    :return: numpy array of string representing the time
    """
    string = []
    for i in range(len(data_list)):
        string.append(data[i].strftime("%m-%d"))
    return np.array(string)



def createModel(company_tuple, learning_total_days):
    """
    Creates the model for each company.
    :param company_tuple: The tuples of company containing company ticks, color of graph, and
        the file name ending with h.5
    :param learning_total_days: The total number of days feeding for the entire model
        has to be greater than 60 as every day uses past 60 days to predict the data
    :return: A list of scaler objects used to inverse transform the predictions and
        A list of x_test varibles used for predictions
    """
    learning_traceback_days = datetime.timedelta(learning_total_days)
    learning_start_date = end_date - learning_traceback_days
    scalers = []
    x_test_list = []
    for i in range(len(company_tuple)):
        print(company_tuple[i][2])
        company = data.DataReader(company_tuple[i][0],
                                  start=learning_start_date,
                                  end=end_date,
                                  data_source='yahoo')['Adj Close']
        prediction_days = 60
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(company.values.reshape((-1, 1)))

        scalers.append(scaler)
        del scaler

        x_train_list = []
        y_train_list = []
        for day in range(prediction_days, len(scaled_data)):
            x_train_list.append(scaled_data[day - prediction_days:day, 0])
            y_train_list.append(scaled_data[day, 0])

        x_train, y_train = np.array(x_train_list), np.array(y_train_list)
        x_test = np.array(x_train_list)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        x_test_list.append(x_test)

        # if the model already exists then don't create anymore
        if (not os.path.isfile(company_tuple[i][2])):
            # need to reshape into 3d for LSTM model to work, the three dimensions represents
            # samples, time steps, and features
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            # y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
            # Start model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            # adding dropout to avoid over fit
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(0.2))
            # no need to return sequences since this is the last layer
            model.add(LSTM(units=50))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, epochs=100, verbose=1)

            # index 2 of the tuple represents the model path
            model.save(company_tuple[i][2])
            del model

    return scalers, x_test_list


if __name__ == '__main__':
    # import font caches
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['font.family'] = 'sans-serif'

    # Company ticks, these companies corresponds to apple, amazon, google, intel, facebook
    companies = [("AAPL", "blue","apple_model.h5"),
                 ("AMZN", "red","amazon_model.h5"),
                 ("GOOGL", "orange", "google_model.h5"),
                 ("INTC", "green", "intel_model.h5"),
                 ("FB", "purple", "facebook_model.h5")]
    end_date = date.today()
    # how many days you want to trace back
    visualization_traceback_days = datetime.timedelta(100)
    visual_start_date = end_date - visualization_traceback_days


    scalers, x_test_list = createModel(companies, 365)

    for i in range(len(companies)):
        plt.figure(i)
        company = data.DataReader(companies[i][0],
                                  start=visual_start_date,
                                  end=end_date,
                                  data_source='yahoo')['Adj Close']
        model = load_model(companies[i][2])
        price = model.predict(x_test_list[i])
        price = scalers[i].inverse_transform(price)

        plt.title(companies[i][0] + " Stock Market")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.xticks(rotation='vertical', fontsize=8)
        plt.plot(convertTimeToString(company.index),
                 company.values, color=companies[i][1], label = companies[i][0] + " Actual Result")
        plt.plot(convertTimeToString(company.index), price[-len(company.index):],
                 color='black', ls='--', label = "Prediction")
        plt.legend()

    plt.show()



