import os
import errno
import pandas as pd
import datetime
from time import time
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from sklearn.metrics import f1_score

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

def get_data_frame(symbol, start_date, end_date, dropna=False, columns=['Date', 'Close'], rename_close=True):
  date_range = pd.date_range(start_date, end_date)
  data_frame = pd.DataFrame(index = date_range)

  symbol_data_frame = pd.read_csv(symbol_to_path(symbol),
      index_col = 'Date',
      parse_dates = True,
      usecols = columns,
      na_values = ['NaN'])

  if rename_close:
      symbol_data_frame = symbol_data_frame.rename(columns = {'Close': symbol})

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

def get_future_price(values, window):
    """Look ahead in the data frame to get the future price after window days"""
    return values.shift(-window)

def preprocess_data(symbol, window, look_ahead, start_date, end_date):
    """Generate new features and labels"""
    df = get_data_frame(symbol, start_date, end_date, dropna=True, columns=['Date', 'Open', 'High', 'Low', 'Close'], rename_close=False)

    df_close = pd.DataFrame(df['Close'])

    # the rolling mean for the window period
    rolling_mean = get_rolling_mean(df_close, window)

    # the Bollinger bands for the window period
    rolling_std = get_rolling_std(df_close, window)
    lower_band, upper_band = get_bollinger_bands(rolling_mean, rolling_std)

    # the cumulative returns for the window period
    cummulative_returns = compute_cummulative_returns(df_close, window)

    # the daily returns
    daily_returns = compute_daily_returns(df_close)

    # future price
    future_price = get_future_price(df_close, look_ahead)

    #rename the columns
    rolling_mean.columns = ['Rolling mean {0}'.format(window)]
    lower_band.columns = ['Lower Bollinger band {0}'.format(window)]
    upper_band.columns = ['Upper Bollinger band {0}'.format(window)]
    cummulative_returns.columns = ['Cummulative return {0}'.format(window)]
    daily_returns.columns = ['Daily return']
    future_price.columns = ['Future Price']

    # so we can join everything into a single dataframe
    df = df.join(daily_returns)
    df = df.join(rolling_mean)
    df = df.join(lower_band)
    df = df.join(upper_band)
    df = df.join(cummulative_returns)
    df = df.join(future_price)

    # calculate the label: 1 if the return compared to the future price is positive, 0 otherwise
    # df['UpDown'] = df.apply(lambda x: 1 if x['Future Price'] - x['Close'] >= 0 else 0, axis=1)

    # keep only the rows that have values for the features calculated on the window
    return df[window:-look_ahead]

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print ("Trained model in {:.4f} seconds".format(end - start))

def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    return f1_score(target.values, y_pred, pos_label=1)

def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print ("Training a {0} using a training set size of {1}...".format(clf.__class__.__name__, len(X_train)))

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    print ("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print ("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))

if __name__ == '__main__':
    start = datetime.datetime(2016,1,1)
    end = datetime.date.today()

    tickers = ['AAPL', 'MSFT', 'GOOG', 'FB', 'AMZN']
    download_data(tickers, start, end)
