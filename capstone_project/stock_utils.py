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

  if(dropna == True):
    return data_frame.dropna()
  else:
    return data_frame

def spy_data_frame(start_date, end_date):
  return get_data_frame('SPY', start_date, end_date, dropna=True)

def get_data_frame_for_symbols(symbols, start_date, end_date, dropspy=False):
  df = spy_data_frame(start_date, end_date)

  for symbol in symbols:
    df = add_symbol_to_data_frame(df, symbol)

  if(dropspy == True):
    df = df.drop('SPY', axis=1)

  return df

def normalize_data(df):
    return df / df.ix[0, :]

def compute_daily_returns(data_frame):
  daily_returns = data_frame.copy()
  daily_returns = data_frame / data_frame.shift(1) - 1
  daily_returns.ix[0,:] = 0
  return daily_returns.fillna(value=0)

def compute_cummulative_returns(data_frame):
  cummulative_returns = data_frame.copy()
  cummulative_returns = (data_frame / data_frame.ix[0,:].values) - 1
  return cummulative_returns

if __name__ == '__main__':
    start = datetime.datetime(2016,1,1)
    end = datetime.date.today()

    tickers = ['AAPL', 'MSFT', 'GOOG', 'FB', 'AMZN']
    download_data(tickers, start, end)
