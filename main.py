from pandas_datareader import data
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


if __name__ == '__main__':
    aapl = data.DataReader("AAPL",
                           start='2020-1-1',
                           end='2020-12-31',
                           data_source='yahoo')['Adj Close']


    print(aapl.to_frame().head())