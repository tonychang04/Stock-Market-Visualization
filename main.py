from datetime import date
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


if __name__ == '__main__':
    # import font caches
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['font.family'] = 'sans-serif'

    # Company ticks, these companies corresponds to apple, amazon, google, intel, facebook
    companies = [("AAPL", "blue"), ("AMZN", "red"), ("GOOGL", "orange"), ("INTC", "green"), ("FB", "purple")]
    end_date = date.today()
    # how many days you want to trace back
    visualization_traceback_days = datetime.timedelta(60)
    visual_start_date = end_date - visualization_traceback_days
    learning_traceback_days = datetime.timedelta(360)
    learning_start_date = end_date - learning_traceback_days

    # temp will delete this later
    company = data.DataReader(companies[0][0],
                              start=learning_start_date,
                              end=end_date,
                              data_source='yahoo')['Adj Close']

    prediction_days = 60
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(company.values.reshape((-1, 1)))

    x_train_list = []
    y_train_list = []

    for i in range(prediction_days, len(scaled_data)):
        x_train_list.append(scaled_data[i - prediction_days:i, 0])
        y_train_list.append(scaled_data[i,0])

    x_train, y_train = np.array(x_train_list), np.array(y_train_list)
    x_test = np.array(x_train_list)
    # need to reshape into 3d for LSTM model to work, the three dimensions represents
    # samples, time steps, and features
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
   # y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    # Start model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
    #adding dropout to avoid over fit
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    # no need to return sequences since this is the last layer
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss= 'mean_squared_error')
    model.fit(x_train, y_train, epochs=100, verbose=1)

    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    price = model.predict(x_test)
    price = scaler.inverse_transform(price)
    plt.plot(convertTimeToString(company.index), company.values, color = 'blue')
    plt.plot(convertTimeToString(company.index[prediction_days:]), price, color = 'red')
    plt.show()



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
