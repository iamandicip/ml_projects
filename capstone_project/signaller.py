class Signaller:
    def __init__(self, config):
        self.upper_limit = config['upper_limit']
        self.lower_limit = config['lower_limit']
        self.rm_tol = config['rm_tol']

    #Trade decisions heuristics

    #calculate trade signal, based on the difference between the rolling mean and the predicted price
    def rm_sig(self, pred_price, rm):
        perc_diff = (pred_price / rm) - 1
        if perc_diff >= self.rm_tol:
            return 1
        elif perc_diff <= -self.rm_tol:
            return -1
        else:
            return 0

    #calculate trade signal, based on the percentage difference between the adjusted close and the predicted price
    def perc_sig(self, adj_close, pred_price):
        perc_diff = (pred_price / adj_close) - 1
        if perc_diff >= self.upper_limit:
            return 1
        elif perc_diff <= -self.lower_limit:
            return -1
        else:
            return 0

    #calculate trade signal, based on the upper and lower bollinger bands, adjusted close and predicted price
    def bb_sig(self, ubb, lbb, adj_close, pred_price):
        if pred_price < adj_close: # sell opportunity?
            if pred_price <= ubb and adj_close >= ubb:
                return -1
        elif pred_price > adj_close: #buy opportunity?
            if pred_price >= lbb and adj_close <= lbb:
                return 1

        return 0

    def calculate_trade_signal(self, x):
        x['RM Signal'] = map(self.rm_sig, x['Predicted Price'], x['Rolling mean 5'])
        x['Percentage Signal'] = map(self.perc_sig, x['Adj. Close'], x['Predicted Price'])
        x['BB Signal'] = map(self.bb_sig, x['Upper Bollinger band 5'], x['Lower Bollinger band 5'], x['Adj. Close'], x['Predicted Price'])

        x['Trade Signal'] = x['RM Signal'] + x['Percentage Signal'] + x['BB Signal']

        return x
