from trader import Trader
import pandas as pd
import numpy as np

class TradingAlgorithm:

    #risk behaviour levels for buying stocks: the higher the value, the higher the confidence that the signal is correct
    buy_risk_levels = {'high' : 1, 'moderate' : 2, 'low' : 3}

    #risk behaviour levels for selling stocks: the lower the value, the higher the confidence that the signal is correct
    sell_risk_levels = {'low' : -3, 'moderate' : -2, 'high' : -1}

    debug_flag = True
    use_signal_strength = False

    trader_configurations = [\
                         {'name' : 'Balanced Trader', \
                          'buy_risk' : buy_risk_levels['moderate'], \
                          'sell_risk' : sell_risk_levels['moderate'], \
                          'use_signal_strength' : use_signal_strength, \
                          'debug' : debug_flag \
                         },\
                         {'name' : 'Bearish Trader', \
                          'buy_risk' : buy_risk_levels['low'], \
                          'sell_risk' : sell_risk_levels['high'], \
                          'use_signal_strength' : use_signal_strength,
                          'debug' : debug_flag \
                         },\
                         {'name' : 'Bullish Trader', \
                          'buy_risk' : buy_risk_levels['high'], \
                          'sell_risk' : sell_risk_levels['low'], \
                          'use_signal_strength' : use_signal_strength, \
                          'debug' : debug_flag \
                         }\
                        ]

    def trade_stock_make_profit(self, config, stock_df, use_predictions):
        if use_predictions:
            trade_df = stock_df[['Adj. Close', 'Trade Signal']]
        else:
            trade_df = stock_df[['Adj. Close', 'Current Signal']]

        trader = Trader(config)
        trader.trade(trade_df)

        return trader.profit

    def trade(self, stocks):
        profits = []

        for s in stocks:
            stock_name = s['name']
            stock_df = s['data']

            print('Trading stock {0}'.format(stock_name))

            no_pred_profits = []
            pred_profits = []

            for config in self.trader_configurations:
                trader_name = config['name']

                config['name'] = trader_name + ' no predictions'
                no_pred_profits.append(self.trade_stock_make_profit(config, stock_df, False))

                config['name'] = trader_name + ' with predictions'
                pred_profits.append(self.trade_stock_make_profit(config, stock_df, True))

                config['name'] = trader_name

            profits_array = np.array([no_pred_profits, pred_profits]).T
            profits_df = pd.DataFrame(profits_array, columns=['Normal Traders', 'Prediction Traders'])

            profits.append({'name' : stock_name, 'profits' : profits_df})

        return profits
