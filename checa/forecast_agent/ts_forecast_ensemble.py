import numpy as np


class TimeSeriesEnsemble:
    def __init__(self, steps_ahead, variable_name, use_pretrained, use_xgboost, use_historical,
                 initial_weights, weight_rate=0.1, error_exponent=1.0):

        self.tau = steps_ahead
        self.variable_name = variable_name
        self.weight_rate = weight_rate
        self.n = error_exponent  # exponent in the error to update the weights
        self.t = 0

        self.use_pretrained = use_pretrained
        if use_pretrained:
            from checa.forecast_agent.pretrained_xgboost import PreTrainedXGBoost
            self.pretrained_xgboost = PreTrainedXGBoost(steps_ahead, variable_name)

        self.use_xgboost = use_xgboost
        if use_xgboost:
            from checa.forecast_agent.online_xgboost import OnlineXGBoost
            self.xgboost = OnlineXGBoost(steps_ahead, variable_name)

        self.use_historical = use_historical
        if use_historical:
            from checa.forecast_agent.historical_predictor import Historical
            self.historical = Historical(steps_ahead, variable_name)

        self.errors_history = {
            'pretrained': [],
            'xgboost': [],
            'historical': []
        }
        self.weights = initial_weights.copy()


    def fit_data(self, hour, data, cur_step):
        """
        data: Numpy array with historical data for the variable.
        variable_name: str
        """
        assert hour == 24, "Error: hour must be 24"
        if self.use_xgboost:
            self.xgboost.fit_data(hour, data, cur_step)


    def predict(self, hour, X, hist_data=None):
        """ X: 1D array with features for predicting variable """

        pretrained_preds = np.zeros(self.tau)
        if self.use_pretrained:
            pretrained_preds = self.pretrained_xgboost.predict(X)

        xgboost_preds = np.zeros(self.tau)
        if self.use_xgboost:
            if self.xgboost.valid_xgboost:
                xgboost_preds = self.xgboost.predict(X)

        historical_preds = np.zeros(self.tau)
        if self.use_historical:
            historical_preds = self.historical.predict(hour, hist_data)

        ensemble_preds = self.get_weighted_average(pretrained_preds, xgboost_preds, historical_preds)

        return ensemble_preds

    def get_weighted_average(self, pretrained_preds, xgboost_preds, historical_preds):
        """
        Get weighted average of predictions from all models.
        """

        # Determine which models have valid predictions.
        valid_pretrained_xgboost = self.use_pretrained
        valid_xgboost = self.use_xgboost and self.xgboost.valid_xgboost
        valid_historical = self.use_historical

        # Adjust weights according to the validity of predictions.
        # This creates a 2D array of weights where each row corresponds to the weights for a specific time step.
        adjusted_weights = np.tile(self.weights, (self.tau, 1))
        if not valid_pretrained_xgboost:
            adjusted_weights[:, 0] = 0
        if not valid_xgboost:
            adjusted_weights[:, 1] = 0
        if not valid_historical:
            adjusted_weights[:, 2] = 0

        # Normalize the weights for each time step.
        total_weights = adjusted_weights.sum(axis=1)
        # Avoid division by zero in case all weights are zero.
        total_weights[total_weights == 0] = 1
        normalized_weights = (adjusted_weights.T / total_weights).T

        # Compute the weighted sum of predictions.
        # Weights are applied to each predictor's predictions and then summed across predictors.
        pred_ensemble = np.einsum('ij,ij->i', normalized_weights,
                                  np.vstack([pretrained_preds, xgboost_preds, historical_preds]).T)

        return pred_ensemble.tolist()


    def update_weights(self, new_value):
        """
        observations: list of lists
        rewards: list of floats
        """

        valid_pretrained = self.use_pretrained
        valid_xgboost = self.use_xgboost and self.xgboost.valid_xgboost
        valid_historical = self.use_historical and self.historical.valid_historical

        if valid_pretrained:
            last_preds_pretrained = self.pretrained_xgboost.last_predictions[0]
            error_pretrained = abs(last_preds_pretrained - new_value)
            self.errors_history['pretrained'].append(error_pretrained)
            avg_error_pretrained = np.mean(self.errors_history['pretrained'])
        else:
            avg_error_pretrained = np.inf

        if valid_xgboost:
            last_preds_xgboost = self.xgboost.last_predictions[0]
            error_xgboost = abs(last_preds_xgboost - new_value)
            self.errors_history['xgboost'].append(error_xgboost)
            avg_error_xgboost = np.mean(self.errors_history['xgboost'])
        else:
            avg_error_xgboost = np.inf

        if valid_historical:
            last_preds_historical = self.historical.last_predictions[0]
            error_historical = abs(last_preds_historical - new_value)
            self.errors_history['historical'].append(error_historical)
            avg_error_historical = np.mean(self.errors_history['historical'])
        else:
            avg_error_historical = np.inf

        best_avg_error = min(avg_error_pretrained, avg_error_xgboost, avg_error_historical)

        if valid_pretrained:
            # remove the first element of the list
            new_w_pretrained = (best_avg_error / avg_error_pretrained)**self.n if avg_error_pretrained > 0 else 1
            self.weights[0] = (1 - self.weight_rate) * self.weights[0] + self.weight_rate * new_w_pretrained
        else:
            new_w_pretrained = np.nan

        if valid_xgboost:
            new_w_xgboost = (best_avg_error / avg_error_xgboost)**self.n if avg_error_xgboost > 0 else 1
            self.weights[1] = (1 - self.weight_rate) * self.weights[1] + self.weight_rate * new_w_xgboost
        else:
            new_w_xgboost = np.nan

        if valid_historical:
            new_w_historical = (best_avg_error / avg_error_historical)**self.n if avg_error_historical > 0 else 1
            self.weights[2] = (1 - self.weight_rate) * self.weights[2] + self.weight_rate * new_w_historical
        else:
            new_w_historical = np.nan

        self.t += 1
        verbose = False
        if verbose and (self.t % 24 == 0 or self.t % 23 == 0):
            print(f"\n{self.t} {self.variable_name}, Avg errors Pretrained XGBoost: {avg_error_pretrained:.4f}, XGBoost: {avg_error_xgboost:.4f}, MAE: Historical: {avg_error_historical:.4f}")
            print(f"Delta Weight: Pretrained XGBoost: {new_w_pretrained:.4f}, XGBoost: {new_w_xgboost:.4f}, Historical: {new_w_historical:.4f}")
            print(f"New weights: {self.weights}")
