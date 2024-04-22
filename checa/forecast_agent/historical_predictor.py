import numpy as np
from xgboost import XGBRegressor


class Historical:
    def __init__(self, steps_ahead, variable_name):
        self.tau = steps_ahead
        self.variable_name = variable_name
        self.last_predictions = np.zeros(self.tau)
        self.valid_historical = False

    def predict(self, hour, hist_data):
        # hours starting at hour and from 0 to 23
        if hour == 24:
            hour = 0
        hours = np.arange(hour, hour + self.tau) % 24
        pred_historical = np.zeros(self.tau)
        for i, h in enumerate(hours):
            if len(hist_data[h]) > 0:
                pred_historical[i] = np.mean(hist_data[h])
                self.valid_historical = True
            else:
                pred_historical[i] = 0
                self.valid_historical = False

        self.last_predictions = pred_historical

        return pred_historical
