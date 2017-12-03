## Files and folders

### Code
- stock_market_data_analisys.ipynb - contains code to download and analyze datasets for stock market data
- stock_market_model_training.ipynb - contains code to find, train, validate and test the appropriate models for the dataset
- stock_market_forecast.ipynb - contains code to generate predictions based on the algorithms found and then simulate trades on historical stock market data based on the predictions
- stock_utils.py - contains utility functions to download the dataset, process the dataset and generate new features
- signaller.py - contains the Signaller class which decides, based on the predictions and the stock data, whether there is a trade signal (buy or sell), and how strong that signal is
- trader.py - contains the Trader class which simulates an agent that does trading based on the stock market data and the signals calculated by the Signaller

### Additional files
- data - contains the csv files with historical stock market data for the selected stocks
- models - contains the pre-trained models
- html - contains the html export from the three jupyter notebooks

## Prerequisites
The environment needs the following libraries installed:
- python 2.7
- pandas 0.19.2
- numpy 1.12.0
- matplotlib 2.0.0
- scikit-learn 0.18.1

## Running the notebooks
The jupyter notebooks must be run in the following order:
1. stock_market_data_analisys.ipynb
2. stock_market_model_training.ipynb
3. stock_market_forecast.ipynb
