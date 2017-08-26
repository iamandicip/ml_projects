import os
import errno
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pandas_datareader.data as web

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def download_data(tickers, start, end):
    source = 'google'

    tickers.append('SPY')

    make_sure_path_exists('data')

    for t in tickers:
        df = web.DataReader(t, source, start, end)
        filename = 'data/{0}.csv'.format(t)

        print('Saving to {0}'.format(filename))

    df.to_csv(filename, encoding='utf-8')

def symbol_to_path(symbol):
  return 'data/{0}.csv'.format(symbol)

def add_symbol_to_data_frame(data_frame, symbol):
  return data_frame.join(
      get_data_frame(symbol, data_frame.index[0], data_frame.index[-1]))

def get_data_frame(symbol, start_date, end_date, dropna=False):
  date_range = pd.date_range(start_date, end_date)
  data_frame = pd.DataFrame(index = date_range)

  symbol_data_frame = pd.read_csv(symbol_to_path(symbol),
      index_col = 'Date',
      parse_dates = True,
      usecols = ['Date', 'Close'],
      na_values = ['NaN'])

  symbol_data_frame = symbol_data_frame.rename(
      columns = {'Close': symbol})

  data_frame = data_frame.join(symbol_data_frame)

  data_frame = fill_missing_values(data_frame)

  if(dropna == True):
    return data_frame.dropna()
  else:
    return data_frame

def spy_data_frame(start_date, end_date):
  return get_data_frame('SPY', start_date, end_date, dropna=True)

def get_data_frame_for_symbols(symbols, start_date, end_date, include_spy=True):
  df = spy_data_frame(start_date, end_date)

  for symbol in symbols:
    df = add_symbol_to_data_frame(df, symbol)

  if(not include_spy):
    df = df.drop('SPY', axis=1)

  return df

def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""
    df_data.fillna(method="ffill", inplace=True)
    df_data.fillna(method="bfill", inplace=True)

    return df_data

def normalize_data(df):
    return df / df.ix[0, :]

def compute_daily_returns(data_frame):
  # daily_returns = data_frame.copy()
  # daily_returns = data_frame / data_frame.shift(1) - 1
  # daily_returns.ix[0,:] = 0
  # return daily_returns.fillna(value=0)
  return compute_cummulative_returns(data_frame, 1)

def compute_cummulative_returns(data_frame, window):
  cummulative_returns = data_frame.copy()
  cummulative_returns = cummulative_returns.pct_change(periods=window)
  return cummulative_returns.fillna(value = 0)

def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    rolling_mean = values.rolling(window=window).mean()

    return rolling_mean

def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return values.rolling(window=window).std()

def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    upper_band = rm + 2 * rstd
    lower_band = rm - 2 * rstd

    return upper_band, lower_band

def preprocess_data(symbol, window, start_date, end_date):
    """Generate new features and labels"""
    df = get_data_frame(symbol, start_date, end_date, dropna=True)

    # the rolling mean for the window period
    rm = get_rolling_mean(df, window)

    # the Bollinger bands for the window period
    rs = get_rolling_std(df, window)
    l_b, u_b = get_bollinger_bands(rm, rs)

    # the cumulative returns for the window period
    c_r = compute_cummulative_returns(df, window)

    # the daily returns
    df = compute_daily_returns(df)

    #rename the columns
    rm.columns = ['Rolling mean']
    l_b.columns = ['Lower Bollinger band']
    u_b.columns = ['Upper Bollinger band']
    c_r.columns = ['Periodic return']
    df.columns = ['Daily return']

    # so we can join everything into a single dataframe
    df = df.join(c_r)
    df = df.join(rm)
    df = df.join(l_b)
    df = df.join(u_b)

    # calculate the label: 1 if the daily return is positive, 0 otherwise
    df['UpDown'] = df.apply(lambda x: 1 if x['Daily return'] >= 0 else 0, axis=1)

    # keep only the rows that have values for the features calculated on the window
    return df[window:]

if __name__ == '__main__':
    start = datetime.datetime(2016,1,1)
    end = datetime.date.today()

    tickers = ['AAPL', 'MSFT', 'GOOG', 'FB', 'AMZN']
    download_data(tickers, start, end)
