class Signaller:
    def __init__(self, config):
        self.delta = config['delta']

    #Trade decisions heuristics

    #calculate trade signal, based on the difference between the rolling mean and the predicted price
    def rm_sig(self, pred_price, rm, adj_close):
        if adj_close <= rm and pred_price >= rm:
            return 1
        elif adj_close >= rm and pred_price <= rm:
            return -1
        else:
            return 0

    #calculate trade signal, based on the percentage difference between the adjusted close and the predicted price
    def perc_sig(self, adj_close, pred_price):
        perc_diff = (pred_price / adj_close) - 1
        if perc_diff >= self.delta:
            return 1
        elif perc_diff <= -self.delta:
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

    def past_rm_sig(self, rm, adj_close):
        perc_diff = (adj_close / rm) - 1
        if perc_diff > self.delta:
            return -1
        elif perc_diff < -self.delta:
            return 1
        else:
            return 0

    def past_bb_sig(self, ubb, lbb, adj_close):
        ubb_perc = (adj_close / ubb) - 1
        lbb_perc = (adj_close / lbb) - 1
        if ubb_perc >= self.delta:
            return -1
        elif lbb_perc <= -self.delta:
            return 1
        else:
            return 0

    def past_adj_close_diff(self, adj_close, past_adj_close):
        perc_diff = (adj_close / past_adj_close) - 1
        if perc_diff >= self.delta:
            return -1
        elif perc_diff <= -self.delta:
            return 1
        else:
            return 0

    def calculate_trade_signal(self, x):
        #first calculate the signals that use the current data only
        x['Current RM Signal'] = map(self.past_rm_sig, x['Rolling mean 5'], x['Adj. Close'])
        x['Current BB Signal'] = map(self.past_bb_sig, x['Upper Bollinger band 5'], x['Lower Bollinger band 5'], x['Adj. Close'])
        x['Past Adj Close Signal'] = map(self.past_adj_close_diff, x['Adj. Close'], x['Past Close 3'])
        x['Current Signal'] = x['Current RM Signal'] + x['Current BB Signal'] + x['Past Adj Close Signal']

        #now calculate the signals that make use of the predictions too
        x['RM Signal'] = map(self.rm_sig, x['Predicted Price'], x['Rolling mean 5'], x['Adj. Close'])
        x['Percentage Signal'] = map(self.perc_sig, x['Adj. Close'], x['Predicted Price'])
        x['BB Signal'] = map(self.bb_sig, x['Upper Bollinger band 5'], x['Lower Bollinger band 5'], x['Adj. Close'], x['Predicted Price'])

        x['Trade Signal'] = x['RM Signal'] + x['Percentage Signal'] + x['BB Signal'] + x['Current Signal']

        return x
