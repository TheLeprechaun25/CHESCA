import numpy as np
from checa.forecast_agent.ts_forecast_ensemble import TimeSeriesEnsemble
from checa.forecast_agent.utils import fast_closest_hour


class ForecastAgent:
    def __init__(self, env, tau, observation_names_b):
        self.tau = tau
        self.env = env
        self.total_steps = 0
        self.n_buildings = len(self.env.buildings_metadata)
        self.observation_names_b = observation_names_b
        self.predicted_variables = ["solar_generation", "outdoor_temp", "dhw_demand", "non_shiftable_load", "occupancy", "cooling_demand", "dhw_usage_bool"]
        self.seen_steps = 0
        self.fitting_frequency = 1  # days
        self.days_until_fit = np.zeros(self.n_buildings)

        # Initialize lists to hold the expected values per hour
        self.hourly_expected_values = {
            'outdoor_temp': [[] for _ in range(24)],
            'solar_generation': [[[] for _ in range(24)] for _ in range(self.n_buildings)],
            'dhw_demand': [[[] for _ in range(24)] for _ in range(self.n_buildings)],
            'non_shiftable_load': [[[] for _ in range(24)] for _ in range(self.n_buildings)],
            'occupancy': [[[] for _ in range(24)] for _ in range(self.n_buildings)],
            'cooling_demand': [[[] for _ in range(24)] for _ in range(self.n_buildings)],
            'dhw_usage_bool': [[[] for _ in range(24)] for _ in range(self.n_buildings)],
            'net_electricity_consumption': [[[] for _ in range(24)] for _ in range(self.n_buildings)],
        }
        self.avg_expected_values = None

        # Initialize Specific datasets for XGBoost
        columns_non_shiftable_load = ['non_shiftable_load', 'cyc_hour1', 'cyc_hour2', 'cyc_day_type1', 'cyc_day_type2', 'exp_load', 'occupancy', 'annual_estimate']
        columns_solar_gen = ['solar_generation', 'cyc_hour1', 'cyc_hour2',  'exp_solar', 'direct_solar', 'diffuse_solar', 'direct_solar_6h', 'diffuse_solar_6h',
                             'direct_solar_12h', 'diffuse_solar_12h', 'direct_solar_24h', 'diffuse_solar_24h']
        columns_dhw_demand = ['dhw_demand', 'cyc_hour1', 'cyc_hour2', 'cyc_day_type1', 'cyc_day_type2', 'exp_dhw_demand', 'occupancy', 'annual_estimate']
        columns_outdoor_temp = ['outdoor_temp', 'cyc_hour1', 'cyc_hour2', 'cyc_day_type1', 'cyc_day_type2', 'direct_solar', 'diffuse_solar', 'exp_outdoor_temp']
        columns_occupancy = ['occupancy', 'cyc_hour1', 'cyc_hour2', 'cyc_day_type1', 'cyc_day_type2', 'exp_occupancy']
        columns_cooling_demand = ['cooling_demand', 'cyc_hour1', 'cyc_hour2', 'exp_cooling_demand', 'temp_out', 'temp_in', 'occupancy']
        columns_dhw_usage_bool = ['dhw_usage_bool', 'cyc_hour1', 'cyc_hour2', 'cyc_day_type1', 'cyc_day_type2', 'exp_dhw_usage_bool', 'occupancy']
        columns_net_electricity_consumption = ['net_electricity_consumption', 'cyc_hour1', 'cyc_hour2', 'cyc_day_type1', 'cyc_day_type2', 'occupancy', 'temp_out', 'temp_in', 'diff_temp_out', 'exp_net_electricity_consumption',
                                               'solar_generation', 'dhw_demand', 'non_shiftable_load', 'cooling_demand']
        self.variable_columns = {
            'non_shiftable_load': columns_non_shiftable_load,
            'solar_generation': columns_solar_gen,
            'dhw_demand': columns_dhw_demand,
            'outdoor_temp': columns_outdoor_temp,
            'occupancy': columns_occupancy,
            'cooling_demand': columns_cooling_demand,
            'dhw_usage_bool': columns_dhw_usage_bool,
            'net_electricity_consumption': columns_net_electricity_consumption
        }

        self.hist_data = {
            'outdoor_temp': [],
            'solar_generation': [[] for _ in range(self.n_buildings)],
            'non_shiftable_load': [[] for _ in range(self.n_buildings)],
            'dhw_demand': [[] for _ in range(self.n_buildings)],
            'occupancy': [[] for _ in range(self.n_buildings)],
            'cooling_demand': [[] for _ in range(self.n_buildings)],
            'dhw_usage_bool': [[] for _ in range(self.n_buildings)],
            'net_electricity_consumption': [[] for _ in range(self.n_buildings)]
        }

        # Initialize prediction models
        self.ensemble_models = {}
        outdoor_temp_args = {
            'steps_ahead': self.tau,
            'variable_name': 'outdoor_temp',
            'use_pretrained': True,
            'use_xgboost': True,
            'use_historical': True,
            'initial_weights': [1.0, 0.0, 0.0],  # pretrained xgboost, online xgboost, historical
            'weight_rate': 0.05,
            'error_exponent': 3.0
        }
        solar_generation_args = {
            'steps_ahead': self.tau,
            'variable_name': 'solar_generation',
            'use_pretrained': False,
            'use_xgboost': True,
            'use_historical': True,
            'initial_weights': [0.0, 0.1, 0.4],
            'weight_rate': 0.05,
            'error_exponent': 3.0
        }
        dhw_demand_args = {
            'steps_ahead': self.tau,
            'variable_name': 'dhw_demand',
            'use_pretrained': True,
            'use_xgboost': True,
            'use_historical': True,
            'initial_weights': [1.0, 0.0, 0.0],
            'weight_rate': 0.05,
            'error_exponent': 3.0
        }
        non_shiftable_load_args = {
            'steps_ahead': self.tau,
            'variable_name': 'non_shiftable_load',
            'use_pretrained': True,
            'use_xgboost': True,
            'use_historical': True,
            'initial_weights': [1.0, 0.0, 0.0],
            'weight_rate': 0.05,
            'error_exponent': 3.0
        }
        self.ensemble_models['outdoor_temp'] = TimeSeriesEnsemble(**outdoor_temp_args)
        self.ensemble_models['solar_generation'] = [TimeSeriesEnsemble(**solar_generation_args) for _ in range(self.n_buildings)]
        #self.ensemble_models['occupancy'] = [TimeSeriesEnsemble(**occupancy_args) for _ in range(self.n_buildings)]
        self.ensemble_models['dhw_demand'] = [TimeSeriesEnsemble(**dhw_demand_args) for _ in range(self.n_buildings)]
        self.ensemble_models['non_shiftable_load'] = [TimeSeriesEnsemble(**non_shiftable_load_args) for _ in range(self.n_buildings)]
        #self.ensemble_models['cooling_demand'] = [TimeSeriesEnsemble() for _ in range(self.n_buildings)]
        #self.ensemble_models['dhw_usage_bool'] = [TimeSeriesEnsemble() for _ in range(self.n_buildings)]

    def compute_forecast(self, observations):
        # Predict the values for the next self.tau hours
        predictions_dict = {b: {} for b in range(self.n_buildings)}

        # Take observation values and save them
        hour = int(observations[self.observation_names_b.index('hour')])

        # Gather past values and save them in historical datasets
        cur_states = self.save_data_and_get_cur_states(observations)

        if self.seen_steps > 0:
            self.update_weights(observations)

        for b in range(self.n_buildings):
            if hour == 24 and self.seen_steps > 0:
                #print(self.days_until_fit[b])
                if self.days_until_fit[b] == 0:
                    self.days_until_fit[b] = self.fitting_frequency - 1
                    self.ensemble_models['solar_generation'][b].fit_data(hour, np.stack(self.hist_data['solar_generation'][b]), self.seen_steps)
                    #self.ensemble_models['occupancy'][b].fit_data(hour, np.stack(self.hist_data['occupancy'][b]), self.seen_steps)
                    self.ensemble_models['dhw_demand'][b].fit_data(hour, np.stack(self.hist_data['dhw_demand'][b]), self.seen_steps)
                    self.ensemble_models['non_shiftable_load'][b].fit_data(hour, np.stack(self.hist_data['non_shiftable_load'][b]), self.seen_steps)

                    if b == 0: # Only compute it once
                        self.ensemble_models['outdoor_temp'].fit_data(hour, np.stack(self.hist_data['outdoor_temp']), self.seen_steps)
                else:
                    self.days_until_fit[b] -= 1

            # Predict the values
            predictions_dict[b]['solar_generation'] = self.ensemble_models['solar_generation'][b].predict(hour, cur_states['solar_generation'][b], self.hourly_expected_values['solar_generation'][b])
            #predictions_dict[b]['occupancy'] = self.ensemble_models['occupancy'][b].predict(hour, cur_states['occupancy'][b], self.hourly_expected_values['occupancy'][b])
            predictions_dict[b]['dhw_demand'] = self.ensemble_models['dhw_demand'][b].predict(hour, cur_states['dhw_demand'][b], self.hourly_expected_values['dhw_demand'][b])
            predictions_dict[b]['non_shiftable_load'] = self.ensemble_models['non_shiftable_load'][b].predict(hour, cur_states['non_shiftable_load'][b], self.hourly_expected_values['non_shiftable_load'][b])
            if b == 0:
                predictions_dict['outdoor_temp'] = self.ensemble_models['outdoor_temp'].predict(hour, cur_states['outdoor_temp'], self.hourly_expected_values['outdoor_temp'])

        self.step()
        return predictions_dict

    def step(self):
        self.seen_steps += 1

    def update_weights(self, observations):
        new_outdoor_temp = observations[self.observation_names_b.index('outdoor_dry_bulb_temperature')]
        self.ensemble_models['outdoor_temp'].update_weights(new_outdoor_temp)

        for b in range(self.n_buildings):
            new_solar_generation = observations[self.observation_names_b.index('solar_generation_' + str(b))]
            #new_occupancy = observations[self.observation_names_b.index('occupant_count_' + str(b))]
            new_dhw_demand = observations[self.observation_names_b.index('dhw_demand_' + str(b))]
            new_non_shiftable_load = observations[self.observation_names_b.index('non_shiftable_load_' + str(b))]

            self.ensemble_models['solar_generation'][b].update_weights(new_solar_generation)
            #self.ensemble_models['occupancy'][b].update_weights(new_occupancy)
            self.ensemble_models['dhw_demand'][b].update_weights(new_dhw_demand)
            self.ensemble_models['non_shiftable_load'][b].update_weights(new_non_shiftable_load)

    def get_expected_values(self, hour):
        if hour == 24:
            used_hour = 0
        else:
            used_hour = hour  # It's hour+1 but -1 for indexing

        expected_values = {
            'outdoor_temp': 0,
            'solar_generation': [0 for _ in range(self.n_buildings)],
            'occupancy': [0 for _ in range(self.n_buildings)],
            'dhw_demand': [0 for _ in range(self.n_buildings)],
            'non_shiftable_load': [0 for _ in range(self.n_buildings)],
            'cooling_demand': [0 for _ in range(self.n_buildings)],
            'dhw_usage_bool': [0 for _ in range(self.n_buildings)],
            'net_electricity_consumption': [0 for _ in range(self.n_buildings)],
        }
        for var in self.predicted_variables:
            if var == 'outdoor_temp':
                if len(self.hourly_expected_values[var][used_hour]) >= 1:
                    expected_values[var] = np.mean(self.hourly_expected_values[var][used_hour])
                else:
                    hours_with_data = [i for i in range(24) if len(self.hourly_expected_values[var][i]) >= 1]
                    closest_hour_with_data = fast_closest_hour(hour, hours_with_data)
                    expected_values[var] = np.mean(self.hourly_expected_values[var][closest_hour_with_data])
            else:
                for b in range(self.n_buildings):
                    if len(self.hourly_expected_values[var][b][used_hour]) >= 1:
                        expected_values[var][b] = np.mean(self.hourly_expected_values[var][b][used_hour])
                    else:
                        hours_with_data = [i for i in range(24) if len(self.hourly_expected_values[var][b][i]) >= 1]
                        closest_hour_with_data = fast_closest_hour(hour, hours_with_data)
                        expected_values[var][b] = np.mean(self.hourly_expected_values[var][b][closest_hour_with_data])

        return expected_values

    def save_data_and_get_cur_states(self, observations):
        """
        Given the observations, save the data and return the current states.
        """

        hour = observations[self.observation_names_b.index("hour")]
        temp_out = observations[self.observation_names_b.index("outdoor_dry_bulb_temperature")]
        cyclical_hour = np.array([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)])
        day_type = observations[self.observation_names_b.index("day_type")]
        cyclical_day_type = np.array([np.sin(2 * np.pi * day_type / 7), np.cos(2 * np.pi * day_type / 7)])
        direct_sol = observations[self.observation_names_b.index("direct_solar_irradiance")]
        diff_sol = observations[self.observation_names_b.index("diffuse_solar_irradiance")]
        direct_sol_6h = observations[self.observation_names_b.index("direct_solar_irradiance_predicted_6h")]
        diff_sol_6h = observations[self.observation_names_b.index("diffuse_solar_irradiance_predicted_6h")]
        direct_sol_12h = observations[self.observation_names_b.index("direct_solar_irradiance_predicted_12h")]
        diff_sol_12h = observations[self.observation_names_b.index("diffuse_solar_irradiance_predicted_12h")]
        direct_sol_24h = observations[self.observation_names_b.index("direct_solar_irradiance_predicted_24h")]
        diff_sol_24h = observations[self.observation_names_b.index("diffuse_solar_irradiance_predicted_24h")]

        self.hourly_expected_values['outdoor_temp'][hour - 1].append(temp_out)

        cur_solar_generation = []
        cur_occupancy = []
        cur_dhw_demand = []
        cur_non_shiftable_load = []
        cur_cooling_demand = []
        cur_dhw_usage_bool = []
        electricity_consumption = []
        for b in range(self.n_buildings):
            cur_solar_generation.append(observations[self.observation_names_b.index("solar_generation_" + str(b))])
            cur_occupancy.append(observations[self.observation_names_b.index("occupant_count_" + str(b))])
            cur_dhw_demand.append(observations[self.observation_names_b.index("dhw_demand_" + str(b))])
            cur_non_shiftable_load.append(observations[self.observation_names_b.index("non_shiftable_load_" + str(b))])
            cur_cooling_demand.append(observations[self.observation_names_b.index("cooling_demand_" + str(b))])
            cur_dhw_usage_bool.append(observations[self.observation_names_b.index("dhw_demand_" + str(b))] > 0.001)
            electricity_consumption.append(observations[self.observation_names_b.index("net_electricity_consumption_" + str(b))])
            self.hourly_expected_values['solar_generation'][b][hour - 1].append(cur_solar_generation[b])
            self.hourly_expected_values['occupancy'][b][hour - 1].append(cur_occupancy[b])
            self.hourly_expected_values['dhw_demand'][b][hour - 1].append(cur_dhw_demand[b])
            self.hourly_expected_values['non_shiftable_load'][b][hour - 1].append(cur_non_shiftable_load[b])
            self.hourly_expected_values['cooling_demand'][b][hour - 1].append(cur_cooling_demand[b])
            self.hourly_expected_values['dhw_usage_bool'][b][hour - 1].append(cur_dhw_usage_bool[b])
            self.hourly_expected_values['net_electricity_consumption'][b][hour - 1].append(electricity_consumption[b])

        # Get expected values and save them. Then get current state for XGBoost
        expected_values = self.get_expected_values(hour)
        self.avg_expected_values = expected_values

        outdoor_temp_X = np.array([temp_out, cyclical_hour[0], cyclical_hour[1], cyclical_day_type[0], cyclical_day_type[1], direct_sol, diff_sol, expected_values['outdoor_temp']])
        self.hist_data['outdoor_temp'].append(outdoor_temp_X)
        solar_generation_X = []
        occupancy_X = []
        dhw_demand_X = []
        non_shiftable_load_X = []
        cooling_demand_X = []
        dhw_usage_bool_X = []
        net_electricity_consumption_X = []
        for b in range(self.n_buildings):
            occupancy = observations[self.observation_names_b.index("occupant_count_" + str(b))]
            temp_in = observations[self.observation_names_b.index("indoor_dry_bulb_temperature_" + str(b))]
            annual_dhw_demand_estimate = self.env.buildings_metadata[b]['annual_dhw_demand_estimate']
            annual_non_shiftable_load_estimate = self.env.buildings_metadata[b]['annual_non_shiftable_load_estimate']

            # Get the current state
            solar_generation_X.append(np.array([cur_solar_generation[b], cyclical_hour[0], cyclical_hour[1], expected_values['solar_generation'][b],
                                                direct_sol, diff_sol, direct_sol_6h, diff_sol_6h, direct_sol_12h, diff_sol_12h, direct_sol_24h, diff_sol_24h]))
            occupancy_X.append(np.array([cur_occupancy[b], cyclical_hour[0], cyclical_hour[1], cyclical_day_type[0], cyclical_day_type[1], expected_values['occupancy'][b]]))
            dhw_demand_X.append(np.array([cur_dhw_demand[b], cyclical_hour[0], cyclical_hour[1], cyclical_day_type[0], cyclical_day_type[1], expected_values['dhw_demand'][b], occupancy, annual_dhw_demand_estimate]))
            non_shiftable_load_X.append(np.array([cur_non_shiftable_load[b], cyclical_hour[0], cyclical_hour[1], cyclical_day_type[0], cyclical_day_type[1], expected_values['non_shiftable_load'][b], occupancy, annual_non_shiftable_load_estimate]))
            cooling_demand_X.append(np.array([cur_cooling_demand[b], cyclical_hour[0], cyclical_hour[1], expected_values['cooling_demand'][b], temp_out, temp_in, occupancy]))
            dhw_usage_bool_X.append(np.array([cur_dhw_usage_bool[b], cyclical_hour[0], cyclical_hour[1], cyclical_day_type[0], cyclical_day_type[1], expected_values['dhw_usage_bool'][b], occupancy]))
            net_electricity_consumption_X.append(np.array([electricity_consumption[b], cyclical_hour[0], cyclical_hour[1], cyclical_day_type[0], cyclical_day_type[1], occupancy, temp_out, temp_in, expected_values['outdoor_temp'] - temp_out,
                                                           expected_values['net_electricity_consumption'][b], cur_solar_generation[b], cur_dhw_demand[b], cur_non_shiftable_load[b], cur_cooling_demand[b]]))

            #
            # Add it to the historical data
            self.hist_data['solar_generation'][b].append(solar_generation_X[b])
            self.hist_data['occupancy'][b].append(occupancy_X[b])
            self.hist_data['dhw_demand'][b].append(dhw_demand_X[b])
            self.hist_data['non_shiftable_load'][b].append(non_shiftable_load_X[b])
            self.hist_data['cooling_demand'][b].append(cooling_demand_X[b])
            self.hist_data['dhw_usage_bool'][b].append(dhw_usage_bool_X[b])
            self.hist_data['net_electricity_consumption'][b].append(net_electricity_consumption_X[b])

        cur_states = {
            'outdoor_temp': outdoor_temp_X,
            'solar_generation': solar_generation_X,
            'occupancy': occupancy_X,
            'dhw_demand': dhw_demand_X,
            'non_shiftable_load': non_shiftable_load_X,
            'cooling_demand': cooling_demand_X,
            'dhw_usage_bool': dhw_usage_bool_X,
            'net_electricity_consumption': net_electricity_consumption_X,
        }
        return cur_states



