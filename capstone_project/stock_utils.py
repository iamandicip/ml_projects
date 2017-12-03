import os
import errno
import pandas as pd
import datetime
from time import time
import io
import requests
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import pylab as pl
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def make_sure_path_exists(path):
    try:
        print('Creating folder {0}'.format(path))
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def download_data(tickers, start, end):
    source = 'google'

    make_sure_path_exists('data')

    for t in tickers:
        df = web.DataReader(t, source, start, end)
        filename = 'data/{0}.csv'.format(t)

        print('Saving to {0}'.format(filename))

        df.to_csv(filename, encoding='utf-8')

def google_stocks(tickers, start, end):

    for t in tickers:
        date_format = '%m+%d+%Y'
        startdate = start.strftime(date_format)

        if not end:
            enddate = time.strftime(date_format)
        else:
            enddate = end.strftime(date_format)

        stock_url = 'http://www.google.com/finance/historical?q=' + t + \
                    '&startdate=' + startdate + '&enddate=' + enddate + '&output=csv'

        raw_response = requests.get(stock_url).content

        filename = 'data/{0}.csv'.format(t)
        with open(filename, 'w') as f:
            print('Saving to {0}'.format(filename))
            f.write(raw_response)

def get_quandl_data(tickers, start, end):
    date_format = '%Y-%m-%d'
    api_key = 'ez-ADzG6c-RPCxB2uzLs'
    base_url = 'https://www.quandl.com/api/v3/datasets/WIKI/{0}/data.csv?start_date={1}&end_date={2}&api_key={3}'
    spy_base_url = 'https://www.quandl.com/api/v3/datatables/ETFG/FUND.csv?ticker={0}&date.gt={1}&date.lt={2}&api_key={3}'

    if not start:
        start = datetime.date(2010, 1, 1)

    if not end:
        end = datetime.date.today()

    make_sure_path_exists('data')

    for t in tickers:
        if t == 'SPY':
            stock_url = spy_base_url.format(t, start.strftime(date_format), end.strftime(date_format), api_key)
        else:
            stock_url = base_url.format(t, start.strftime(date_format), end.strftime(date_format), api_key)

        print(stock_url)

        raw_response = requests.get(stock_url).content

        filename = 'data/{0}.csv'.format(t)
        with open(filename, 'w') as f:
            print('Saving to {0}'.format(filename))
            f.write(raw_response)


def symbol_to_path(symbol):
  return 'data/{0}.csv'.format(symbol)

def add_symbol_to_data_frame(data_frame, symbol):
  return data_frame.join(
      get_data_frame(symbol, data_frame.index[0], data_frame.index[-1]))

def get_data_frame(symbol, start_date, end_date, dropna=False, columns=['Date', 'Adj. Close'], rename_close=True):
  date_range = pd.date_range(start_date, end_date)
  data_frame = pd.DataFrame(index = date_range)

  symbol_data_frame = pd.read_csv(symbol_to_path(symbol),
      index_col = 'Date',
      parse_dates = True,
      usecols = columns,
      na_values = ['NaN'])

  if rename_close:
      symbol_data_frame = symbol_data_frame.rename(columns = {'Adj. Close': symbol})

  data_frame = data_frame.join(symbol_data_frame)

  data_frame = fill_missing_values(data_frame)

  if(dropna == True):
    return data_frame.dropna()
  else:
    return data_frame

def spy_data_frame(start_date, end_date):
  return get_data_frame('SPY', start_date, end_date, dropna=True)

def get_data_frame_for_symbols(symbols, start_date, end_date, include_spy=False):
  df = None

  for symbol in symbols:
    if df is None:
        df = get_data_frame(symbol, start_date, end_date)
    else:
        df = add_symbol_to_data_frame(df, symbol)

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
    df = get_data_frame(symbol, start_date, end_date, dropna=True, columns=['Date', 'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close'], rename_close=False)

    df_close = pd.DataFrame(df['Adj. Close'])

    # the rolling mean for the window period
    rolling_mean = get_rolling_mean(df_close, window)

    # the Bollinger bands for the window period
    rolling_std = get_rolling_std(df_close, window)
    upper_band, lower_band = get_bollinger_bands(rolling_mean, rolling_std)

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

def plot_predictions(title, y, y_hat):
    pl.title(title)
    pl.plot(y, y_hat, 'ro')
    pl.plot([np.amin(y), np.amax(y)],[np.amin(y_hat), np.amax(y_hat)], 'g-')
    pl.xlabel('Real values')
    pl.ylabel('Predicted values')
    pl.show()

def print_cross_val_accuracy(est, X, y):
    scores = cross_val_score(est, X, y)
    print('cross validation accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))

def cross_val_splits(X, n_splits=5):
    splits = TimeSeriesSplit(n_splits)
    cv_splits = []

    for train_index, test_index in splits.split(X):
        cv_splits.append((train_index, test_index))

    return cv_splits

def calculate_predictions_for_dataset(dataset, estimators, plot_and_print=True):
    data = dataset.drop(labels=['Future Price'], axis=1)
    target = dataset['Future Price']

    predictions = []

    for reg in estimators:
        class_name = reg.named_steps['reg'].__class__.__name__

        if plot_and_print:
            print ('\n{0}: \n'.format(class_name))

        pred = reg.predict(data)

        score = reg.score(data, target)

        if plot_and_print:
            print('score for validation set: {0}'.format(score))
            plot_predictions(class_name, target, pred)

        predictions.append(pred)

    average_predictions = np.zeros(target.shape[0])

    for p in predictions:
        average_predictions += p

    average_predictions /= len(predictions)

    if plot_and_print:
        print('\nr2 score for average predictions: {0}'.format(r2_score(target, average_predictions)))
        print('\nmean squared error for average predictions: {0}'.format(mean_squared_error(target, average_predictions)))

        plot_predictions('Average predictions', target, average_predictions)

    return average_predictions

if __name__ == '__main__':
    start = datetime.datetime(2010,1,1)
    end = datetime.date.today()

    tickers = ['AAPL', 'XOM']
    # download_data(tickers, start, end)
    get_quandl_data(tickers, start, end)
