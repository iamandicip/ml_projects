class Trader:
    def __init__(self, config):
        self.name = config['name']
        self.initial_funds = config['initial_funds']
        self.funds = config['initial_funds']
        self.buy_risk = config['buy_risk']
        self.sell_risk = config['sell_risk']
        self.use_signal_strength = config['use_signal_strength']
        self.min_stocks = 100
        self.stocks = 0
        #transaction fees in percentage
        self.transaction_fees = 0
        #number of transactions made by the trader
        self.transactions = 0
        self.profit = 0

        self.debug_flag = config['debug']

        print('{0} starts trading'.format(self.name))

    def debug(self, message):
        if self.debug_flag:
            print(message)

    #sell everything at the end of the period
    def exit_position(self, last_adj_close):
        self.funds = self.funds + self.stocks * last_adj_close * (1 - self.transaction_fees)
        return ((self.funds - self.initial_funds) / self.initial_funds) * 100

    def calculate_stocks_to_trade(self, signal):
        traded_stock = 0

        make_trade = (signal > 0 and signal >= self.buy_risk) or (signal < 0 and signal <= self.sell_risk)
        trade_direction = lambda x: (1, -1)[x < 0]

        if make_trade:
            if self.use_signal_strength:
                traded_stock = signal * self.min_stocks
            else:
                traded_stock = trade_direction(signal) * self.min_stocks

        return traded_stock

    def trade(self, df):
        first_adj_close = 0
        last_adj_close = 0
        failed_tries = 0
        for row in df.itertuples():
            date = row[0]
            adj_close = row[1]
            last_adj_close = row[1]
            signal = row[2]
            #cache the first adjusted close
            if first_adj_close == 0:
                first_adj_close = row[1]

            stocks_to_trade = self.calculate_stocks_to_trade(signal)
            cash_required = adj_close * stocks_to_trade * (1 + self.transaction_fees)

            buy_decision = stocks_to_trade > 0
            sell_decision = stocks_to_trade < 0

            #buy decision
            if buy_decision:
                #first check if trader has enough funds to buy the stocks
                if self.funds >= cash_required:
                    self.debug('{0} buys {1} stocks for {2} $ on {3}'.format(self.name, stocks_to_trade, cash_required, date))
                    self.stocks += stocks_to_trade
                    self.funds -= cash_required
                    self.transactions += 1
                else:
                    self.debug('{0} doesn''t have enough funds to buy the stocks!'.format(self.name))

            #sell decision
            elif sell_decision:
                #firs check if trader has enough stocks to sell
                if self.stocks >= -stocks_to_trade:
                    self.debug('{0} sells {1} stocks for {2} $ on {3}'.format(self.name, -stocks_to_trade, -cash_required, date))
                    self.stocks += stocks_to_trade
                    self.funds -= cash_required
                    self.transactions += 1
                else:
                    self.debug('{0} doesn''t have enough stocks to sell!'.format(self.name))

        self.debug('First vs Last adjusted close: {0} vs {1}'.format(first_adj_close, last_adj_close))
        self.profit = self.exit_position(last_adj_close)

        print('After {0} transactions, {1} exits his position with a profit of {2:.2f}%\n'.format(self.transactions, self.name, self.profit))
