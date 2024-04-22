import numpy as np
from xgboost import XGBRegressor
#from sklearn.model_selection import train_test_split


class OnlineXGBoost:
    def __init__(self, steps_ahead, variable_name):
        self.tau = steps_ahead
        self.variable_name = variable_name
        self.t = 0

        xgboost_params = {
            'n_estimators': 75,
            'max_depth': 3,
            'learning_rate': 0.1,
            'verbosity': 0,
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'booster': 'gbtree',
            'n_jobs': -1,
            'random_state': 0,
        }
        self.models_xgboost = [XGBRegressor(**xgboost_params) for _ in range(steps_ahead)]
        self.valid_xgboost = False

        self.last_predictions = np.zeros(self.tau)


    def fit_data(self, hour, data, cur_step):
        """
        data: Numpy array with historical data for the variable.
        variable_name: str
        """
        assert hour == 24, "Error: hour must be 24"  # TODO: start at 1


        # Fit XGBoost models with new data. For each step ahead, fit a new model.
        if cur_step < self.tau+5:
            # No data for i-th step ahead
            self.valid_xgboost = False
        else:
            self.valid_xgboost = True
            # Variable value is in col 0
            #data1, data2 = train_test_split(data, train_size=0.99)

            y_train = data[:cur_step + 1, 0]
            X_train = data[:cur_step + 1, :]
            for i in range(self.tau):
                y_train_step = np.roll(y_train, -(i+1))[:-(i+1)]
                X_train_step = X_train[:-(i+1)]
                # Fit XGBoost model. TODO: split train/test?

                self.models_xgboost[i].fit(X_train_step, y_train_step) #, eval_set=[(X_test_step, y_test_step)], verbose=False)

    def predict(self, X):
        """ X: 1D array with features for predicting variable """
        assert self.valid_xgboost, "Error: XGBoost models have not been fitted yet."
        # Predict with XGBoost Online trained models
        pred_xgboost = np.zeros(self.tau)
        for i in range(self.tau):
            pred_xgboost[i] = self.models_xgboost[i].predict(X[np.newaxis, :])
            # make 0 the negative values
            pred_xgboost[pred_xgboost < 0] = 0

        self.last_predictions = pred_xgboost
        return pred_xgboost
