from datetime import date
from pandas_datareader import data
from sklearn.preprocessing import MinMaxScaler

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
    start_date = end_date - visualization_traceback_days
    #temp will delete this later
    company = data.DataReader(companies[0][0],
                    start=start_date,
                    end=end_date,
                    data_source='yahoo')['Adj Close']
    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_data = scaler.fit_transform(company.values.reshape((-1,1)))
    print(scaled_data[:,:])

    figure, (price_plot, percent_change_plot) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    for company in companies:
        company_series = data.DataReader(company[0],
                                         start=start_date,
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

