import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import os

def get_data(tickers, start_date, end_date):
    data = 'etfs_data_clean.csv'
    #Check if data exists already, if not download and clean:
    if os.path.exists('etfs_data_clean.csv'):
        print('Loading data from local file...')
        prices_df = pd.read_csv('etfs_data_clean.csv', index_col = 0, parse_dates = True)
    else:
        print('Downloading data from yahoo finance...')
        prices_df_raw = yf.download(tickers, start = start_date, end = end_date, interval = '1d', auto_adjust = False)['Adj Close'] 
        #With auto-adjust not turned off the closing price is already adjusted for splits and dividends and one can use just the 'Close' column, but I left it here for clarity
        #Replace any NA value:
        prices_df_clean = prices_df_raw.fillna(method = 'ffill')
        #If there are still NAs it must be at the beginning of the file so:
        if prices_df_clean.isnull().sum().any():
            prices_df_clean = prices_df_clean.fillna(method='bfill')
        #Check date and time format and make sure data is sorted by date:
        if not isinstance(prices_df_clean.index, pd.DatetimeIndex):
            prices_df_clean.index = pd.to_datetime(prices_df_clean.index)
        prices_df_clean = prices_df_clean.sort_index()
        prices_df_clean.to_csv('etfs_data_clean.csv')
        prices_df = prices_df_clean
        print('Data saved to etfs_data_clean.csv')
    return prices_df

def quarterly_rebalancing(returns_df, div_tickers, target_weights, initial_capital):
    quarterly_dates = returns_df.resample('QE').last().index #ending dates of each quarter
    values_df = pd.DataFrame(index = returns_df.index, columns = div_tickers) #storing each ETF value over time separetely to adjust weights later
    values_df.iloc[0] = initial_capital * target_weights

    for i in range(1, len(returns_df)):
        current_date = returns_df.index[i]
        values_df.iloc[i] = values_df.iloc[i-1] * (1 + returns_df[div_tickers].iloc[i]) #adjust each ETF value simultaneously

        #rebalance on quarter-end:
        if current_date in quarterly_dates:
            total_value = values_df.iloc[i].sum()
            values_df.iloc[i] = total_value * target_weights

    #total portfolio value is the sum of all holdings each day
    div_equity = values_df.sum(axis=1)
    div_returns = div_equity.pct_change().fillna(0).astype(float) #this is so taht pandas knows in the future the data type, cause it will turn off automatic recognition for .fillna function
    return div_equity, div_returns

def downside_deviation(window, MAR):
    excess_returns = window - MAR
    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.empty:
        return 0
    return downside_returns.std()