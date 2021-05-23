from pandas_datareader import data
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


if __name__ == '__main__':
    companies = []
    aapl = data.DataReader("AAPL",
                           start='2020-1-1',
                           end='2020-12-31',
                           data_source='yahoo')['Adj Close']

    print(aapl.index)

    plt.plot(aapl.index, aapl.values)
    plt.show(block=True)
    plt.interactive(True)