from pandas_datareader import data
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


if __name__ == '__main__':
    # Company ticks, these companies corresponds to apple, amazon, google, intel, facebook
    companies = [("AAPL","blue"),("AMZN","red"), ("GOOGL","orange") ,("INTC","green"),("FB","purple")]
    start_date = '2020-1-1'
    end_date = '2020-12-31'
    for company in companies:
        company_series= data.DataReader(company[0],
                           start = start_date,
                           end = end_date,
                           data_source='yahoo')['Adj Close']
        plt.plot(company_series.index, company_series.values, color = company[1])


    #plt.plot(aapl.index, aapl.values)
    plt.legend([company[0] for company in companies])
    plt.title("Closed Stock Price Per Day")
    plt.xlabel('Date')
    plt.ylabel('Price Per Stock')
    plt.show(block=True)
   # plt.interactive(True)