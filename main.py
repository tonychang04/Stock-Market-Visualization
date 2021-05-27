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


def convertTimeToString(data):
    string = []
    for i in range(len(data)):
        string.append(data[i].strftime("%m-%d"))
    return np.array(string)

def createModel(company_tuple, learning_total_days):
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

        x_train_list = []
        y_train_list = []

        for day in range(prediction_days, len(scaled_data)):
            x_train_list.append(scaled_data[day - prediction_days:day, 0])
            y_train_list.append(scaled_data[day, 0])

        x_train, y_train = np.array(x_train_list), np.array(y_train_list)
        x_test = np.array(x_train_list)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        x_test_list.append(x_test)
        """
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
        """
        scalers.append(scaler)
        del scaler
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
    visualization_traceback_days = datetime.timedelta(150)
    visual_start_date = end_date - visualization_traceback_days

    print(companies[0][2])
    scalers, x_test_list = createModel(companies, 365)

    figure, ax = plt.subplots(len(companies), 1, sharex=True, figsize=(12, 8))
    for i in range(len(companies)):
        company = data.DataReader(companies[i][0],
                                  start=visual_start_date,
                                  end=end_date,
                                  data_source='yahoo')['Adj Close']
        model = load_model(companies[i][2])
        price = model.predict(x_test_list[i])
        price = scalers[i].inverse_transform(price)
        ax[i].title(companies[i][0])
        ax[i].plot(convertTimeToString(company.index), company.values, color= companies[i][1])
        ax[i].plot(convertTimeToString(company.index), price[-len(company.index):], color='black', ls = '--')
        ax[i].grid(True)


    plt.xticks(rotation='vertical', fontsize=8)
    figure.tight_layout()
    plt.show()



    """
    price = model.predict(x_test)
    price = scaler.inverse_transform(price)
    plt.plot(convertTimeToString(company.index), company.values, color = 'blue')
    plt.plot(convertTimeToString(company.index[prediction_days:]), price, color = 'red')
    plt.show()
    """



    """
    figure, (price_plot, percent_change_plot) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    for company in companies:
        company_series = data.DataReader(company[0],
                                         start=visual_start_date,
                                         end=end_date,
                                         data_source='yahoo')['Adj Close']
        price_plot.plot(convertTimeToString(company_series.index), company_series.values, color=company[1])
        percent_change_plot.plot(convertTimeToString(company_series.index), company_series.pct_change(),
                                 color=company[1])

    plt.xlabel('Date')
    plt.xticks(rotation='vertical', fontsize=11)
    figure.tight_layout()
    figure.legend([company[0] for company in companies])

    price_plot.set_ylabel('Price Per Stock')
    price_plot.grid(True)
    percent_change_plot.set_ylabel('Price Percent Change')
    percent_change_plot.grid(True)

    plt.show()
    """
