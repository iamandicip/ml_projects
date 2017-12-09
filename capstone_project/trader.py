class Trader:

    #funds available with which the trader starts
    initial_funds = 100000

    #number of stocks with which the trader stards
    initial_stocks = 500

    #minimum amount of stocks that can be traded
    min_stocks = 100

    #transaction fees in percentage
    transaction_fees = 0.001

    def __init__(self, config):
        self.name = config['name']
        self.buy_risk = config['buy_risk']
        self.sell_risk = config['sell_risk']
        self.use_signal_strength = config['use_signal_strength']
        self.stocks = self.initial_stocks
        self.funds = self.initial_funds

        #number of transactions made by the trader
        self.transactions = 0
        self.profit = 0

        self.cash_made = 0
        self.cash_spent = 0

        self.debug_flag = config['debug']

        print('{0} starts trading'.format(self.name))

    def debug(self, message):
        if self.debug_flag:
            print(message)

    #enter the position by buying stock worth of half the initial funds
    # def enter_position(self, first_adj_close):
    #     cash_to_spend = self.initial_funds * self.stock_funds_ratio
    #     raw_stocks_to_buy, change = divmod(cash_to_spend, first_adj_close)
    #     stocks_to_buy, change = divmod(raw_stocks_to_buy, self.min_stocks)
    #
    #     stocks_to_buy = stocks_to_buy * self.min_stocks
    #
    #     spent_cash = stocks_to_buy * first_adj_close * (1 + self.transaction_fees)
    #     self.funds -= spent_cash
    #     self.stocks = stocks_to_buy
    #
    #     print('{0} enters position by buying {1} stocks at {2:.2f}$ and has {3:.2f}$ left to invest'.format(self.name, self.stocks, first_adj_close, self.funds))

    #we consider the intial stocks value as invested cash
    def start_trading(self, first_adj_close):
        self.cash_spent = self.stocks * first_adj_close * (1 - self.transaction_fees)

    #sell everything at the end of the period
    def end_trading(self, last_adj_close):
        if self.transactions > 0:
            self.cash_made += self.stocks * last_adj_close * (1 - self.transaction_fees)
            #return profit in percentage
            return ((self.cash_made / self.cash_spent) - 1) * 100
        else:
            #if the trader didn't make any trade decisions, then the profit from the initial stocks is irrelevant
            return 0

    def calculate_stocks_to_trade(self, signal):
        traded_stock = 0

        make_trade = (signal > 0 and signal >= self.buy_risk) or (signal < 0 and signal <= self.sell_risk)
        trade_direction = lambda x: (1, -1)[x < 0]

        if make_trade:
            if self.use_signal_strength:
                traded_stock = signal * self.min_stocks
            else:
                #return negative stock for a sell order
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
                #start trading
                self.start_trading(first_adj_close)

            stocks_to_trade = self.calculate_stocks_to_trade(signal)

            cash_spent = adj_close * abs(stocks_to_trade) * (1 + self.transaction_fees)
            cash_received = adj_close * abs(stocks_to_trade) * (1 - self.transaction_fees)

            buy_decision = stocks_to_trade > 0
            sell_decision = stocks_to_trade < 0

            #buy decision
            if buy_decision:
                #first check if trader has enough funds to buy the stocks
                if self.funds >= cash_spent:
                    self.debug('{0} buys {1} stocks for {2:.2f} $ on {3}'.format(self.name, stocks_to_trade, cash_spent, date))
                    self.stocks += stocks_to_trade
                    self.funds -= cash_spent
                    self.cash_spent += cash_spent
                    self.transactions += 1
                # else:
                    # self.debug('{0} doesn''t have enough funds to buy the stocks!'.format(self.name))

            #sell decision
            elif sell_decision:
                #firs check if trader has enough stocks to sell
                if self.stocks >= abs(stocks_to_trade):
                    self.debug('{0} sells {1} stocks for {2:.2f} $ on {3}'.format(self.name, -stocks_to_trade, cash_received, date))
                    self.stocks -= abs(stocks_to_trade)
                    self.funds += cash_received
                    self.cash_made += cash_received
                    self.transactions += 1
                # else:
                    # self.debug('{0} doesn''t have enough stocks to sell!'.format(self.name))

        # self.debug('First vs Last adjusted close: {0:.2f} vs {1:.2f}'.format(first_adj_close, last_adj_close))

        print('After {0} transactions, {1} has {2} stocks and {3:.2f}$ cash spent and {4:.2f}$ cash made'\
              .format(self.transactions, self.name, self.stocks, self.cash_spent, self.cash_made))

        self.profit = self.end_trading(last_adj_close)

        print('{1} ends trading with a profit of {2:.2f}%\n'.format(self.transactions, self.name, self.profit))

        # cumm_return = (float)((last_adj_close / first_adj_close) - 1) * 100
        # print('For the same period, this particular stock cummulative return is {0:.2f}%\n'.format(cumm_return))
