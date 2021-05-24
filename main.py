from pandas_datareader import data
import matplotlib.pyplot as plt
from datetime import date
import datetime


import numpy as np

if __name__ == '__main__':
    # Company ticks, these companies corresponds to apple, amazon, google, intel, facebook
    companies = [("AAPL", "blue"), ("AMZN", "red"), ("GOOGL", "orange"), ("INTC", "green"), ("FB", "purple")]
    end_date = date.today()
    # how many days you want to trace back
    traceback_days = datetime.timedelta(90)
    start_date = end_date - traceback_days
    figure, (price_plot, percent_change_plot) = plt.subplots(2, 1, sharex=True, figsize = (12,8))

    for company in companies:
        company_series = data.DataReader(company[0],
                                         start=start_date,
                                         end=end_date,
                                         data_source='yahoo')['Adj Close']
        price_plot.plot(company_series.index, company_series.values, color=company[1])
        percent_change_plot.plot(company_series.index, company_series.pct_change(), color=company[1])

    plt.suptitle("Stock Data")
    figure.legend([company[0] for company in companies])
    plt.xlabel('Date')
    figure.tight_layout()
    price_plot.set_ylabel('Price Per Stock')
    percent_change_plot.set_ylabel('Price Percent Change')
    plt.show()
