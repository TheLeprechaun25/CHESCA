import numpy as np
from typing import List
from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv
from checa.cooling_device_controller.cooling_device_controller import CoolingDeviceController
from checa.forecast_agent.forecasting_agent import ForecastAgent
from checa.battery_control_search.battery_controller import BatteryController
from checa.utils import get_observation_names_with_building


class Checa(Agent):
    def __init__(self, env: CityLearnEnv, params=None, **kwargs):
        super().__init__(env, **kwargs)
        self.env = env
        self.n_buildings = len(self.env.buildings_metadata)
        self.observation_names = env.observation_names
        self.observation_names_b = get_observation_names_with_building(env.observation_names[0])
        self.seen_steps = 0

        if params is not None:
            self.params = params
        else:
            min_soc_per_hour = {
                "0": 0.60,
                "1": 0.65,  # Assuming a consistent high SoC through the night
                "2": 0.72,
                "3": 0.78,
                "4": 0.80,  # This in reality is 5
                "5": 0.85,
                "6": 0.80,  # Decrease as energy usage increases in the morning
                "7": 0.75,
                "8": 0.70,
                "9": 0.60,
                "10": 0.50,  # Midday when solar might be charging the battery
                "11": 0.60,
                "12": 0.65,
                "13": 0.65,
                "14": 0.70,
                "15": 0.70,  # Increasing SoC as solar generation remains high but usage increases
                "16": 0.70,
                "17": 0.65,
                "18": 0.70,  # Evening when energy usage is high
                "19": 0.60,
                "20": 0.60,
                "21": 0.60,  # Preparing for the lower usage overnight
                "22": 0.60,
                "23": 0.55,
            }

            self.params = {
                'tau': 1,  # Number of future time steps to consider
                'balance_type': 'C', # A B or C - Defines how to compute fitness in battery action tree search. C seems best
                'dt': 0.05,  # action discretisation step for battery action (soc)

                'max_soc_normal': 0.99,  # Max battery soc when there is no outage. Close to 0.8 is good for outage when large demand, to be able to discharge large quantities
                'min_soc_per_hour': min_soc_per_hour,  # Min battery soc per hour when there is no outage
                'max_soc_outage': 0.87,  # Max battery soc when there is an outage
                'B_low': 1.18,  # Parameter to tune: how much we want to be sure we are not consuming more or less than avg consumption
                'B_high': 1.0,

                'TMP_max_reduction_percent': 0.0,  # Max percentage of reduction of the TMP action when there is need to reduce total consumption
                'peak_TMP_reduction': 0.0, #1,  # Additional reduction to tmp whenever there might be a peak

                'max_soc_reduction_in_outage': 0.70, #0.4766106595008137, #0.30,  # Max reduction of battery soc when there is an outage
                'increase_scale_outage': 1.166, # 1.1,
            }

        # Initialize Predictor Agent
        self.forecast_agent = ForecastAgent(env, self.params['tau'], self.observation_names_b)
        self.forecasts = None

        # Initialize variables used by the central agent
        self.elec_consumption_history = [[] for _ in range(self.n_buildings)]
        self.elec_consumption_history_per_hour = [[[] for _ in range(24)] for _ in range(self.n_buildings)]
        self.elec_consumption_prediction_history = [[] for _ in range(self.n_buildings)]

        self.predicted_dhw_device_demand = [0.0 for _ in range(self.n_buildings)]

        self.predicted_cooling_demand = [0.0 for _ in range(self.n_buildings)]
        self.predicted_battery_demand = [0.0 for _ in range(self.n_buildings)]

        self.future_dhw_demands = None
        self.future_cooling_demands = None

        # Cooling device controller
        self.cooling_device_controller = [CoolingDeviceController(self.env.buildings_metadata[b], b, self.observation_names_b) for b in range(self.n_buildings)]

        # Battery features and important hyperparameters
        self.battery_capacities = np.array([self.building_metadata[b]['electrical_storage']['capacity'] for b in range(self.n_buildings)])
        self.battery_nominal_powers = np.array([self.building_metadata[b]['electrical_storage']['nominal_power'] for b in range(self.n_buildings)])
        self.battery_efficiency_curves = np.array([self.building_metadata[b]['electrical_storage']['power_efficiency_curve'] for b in range(self.n_buildings)])
        self.battery_capacity_power_curves = np.array([self.building_metadata[b]['electrical_storage']['capacity_power_curve'] for b in range(self.n_buildings)])
        self.min_battery_soc = np.array([1 - self.building_metadata[b]['electrical_storage']['depth_of_discharge'] for b in range(self.n_buildings)])
        self.cur_battery_soc = np.array([0.0 for _ in range(self.n_buildings)])

        # Initialize Battery Controller
        self.battery_controller = [BatteryController(self.battery_capacities[b], self.params['tau'], self.params['min_soc_per_hour'], self.params['dt'], self.params['max_soc_normal'], self.params['balance_type']) for b in range(self.n_buildings)]

        # DHW device/storage features
        self.dhw_storage_capacity = np.array([self.building_metadata[b]['dhw_storage']['capacity'] for b in range(self.n_buildings)])
        self.dhw_device_efficiency = np.array([self.building_metadata[b]['dhw_device']['efficiency'] for b in range(self.n_buildings)])
        self.dhw_device_nominal_powers = np.array([self.building_metadata[b]['dhw_device']['nominal_power'] for b in range(self.n_buildings)])
        self.cur_dhw_soc = np.array([0.0 for _ in range(self.n_buildings)])

        # put 0.2 at hours 2, 3, 4, 5, 14 in an array of shape (n_buildings, 24)
        self.cur_dhw_heat_schedules = np.zeros((self.n_buildings, 24))
        self.cur_dhw_heat_schedules[:, 2:6] = 0.2
        self.cur_dhw_heat_schedules[:, 14] = 0.2

        self.outage_details = [{
            'outage_duration': 0,
            'time_since_last_outage': 0,
            'saturation_perc': False,
            'outage_flag': False,
            'curr_error': 0,
            'outage_previously': False,
            'saturated_previously': False,
        } for _ in range(self.n_buildings)]

        self.plot = False

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        assert len(observations) == 1, "Only central agent is supported"
        observations = observations[0]
        hour = int(observations[self.observation_names_b.index('hour')])

        # Add observations to forecast agent and get forecasts
        self.forecasts = self.forecast_agent.compute_forecast(observations)

        # COMPUTE ACTIONS: Each building independently
        action_proposals = self.initial_actions(observations)

        self.future_cooling_demands = self.get_future_cooling_demands(observations, action_proposals)
        self.future_dhw_demands = self.get_future_dhw_demands(observations)

        # REFINE AGENT ACTIONS: IMPROVE RAMPING AND GRID IN GENERAL
        # Only refine actions after the first step
        if self.seen_steps >= 1:
            action_proposals = self.refine_actions_with_battery_controller(hour, action_proposals)
        self.compute_elec_consumption_values()

        # Make a step
        self.step()

        return [action_proposals]

    def step(self):
        self.seen_steps += 1

    def initial_actions(self, observations):
        hour = int(observations[self.observation_names_b.index('hour')])
        predicted_outdoor_temp = self.forecasts['outdoor_temp'][0]

        # COMPUTE ACTIONS: Each building independently
        action_proposals = []
        for b in range(self.n_buildings):
            # Save some variables used
            self.cooling_device_controller[b].get_cop(predicted_outdoor_temp)
            self.outage_details[b]['outage_flag'] = observations[self.observation_names_b.index('power_outage_' + str(b))] == 1
            self.cur_battery_soc[b] = observations[self.observation_names_b.index('electrical_storage_soc_' + str(b))]
            self.cur_dhw_soc[b] = observations[self.observation_names_b.index('dhw_storage_soc_' + str(b))]
            net_electricity_consumption = observations[self.observation_names_b.index(f'net_electricity_consumption_' + str(b))]
            self.elec_consumption_history[b].append(net_electricity_consumption)
            self.elec_consumption_history_per_hour[b][hour-1].append(net_electricity_consumption)
            last_step_cooling_demand = observations[self.observation_names_b.index(f'cooling_demand_' + str(b))] if self.seen_steps > 13 else 0.0

            # Separate actions for OUTAGE and NORMAL operation
            if self.outage_details[b]['outage_flag']:
                self.outage_details[b]['outage_previously'] = True
                self.outage_details[b]['time_since_last_outage'] = 0
                self.outage_details[b]['outage_duration'] += 1

                available_soc = max(float(self.cur_battery_soc[b] - self.min_battery_soc[b]), 0.0)
                max_soc_reduction = min(self.params['max_soc_reduction_in_outage'], available_soc)
                max_elec_in_battery = self.battery_capacities[b] * max_soc_reduction

                # 1) TMP action during outage
                expected_available_elec = max_elec_in_battery + self.forecasts[b]['solar_generation'][0]
                TMP_action = self.cooling_device_controller[b].find_best_action(observations, predicted_outdoor_temp, last_step_cooling_demand, self.outage_details[b], expected_available_elec)
                cooling_elec_demand, cooling_energy_demand = self.cooling_device_controller[b].compute_pred_cooling_consumption(TMP_action, predicted_outdoor_temp)

                # 2) DHW action during outage --> default always discharging
                if self.forecasts[b]['dhw_demand'][0] < self.cur_dhw_soc[b] * self.dhw_storage_capacity[b]:
                    dhw_demand = 0.0
                else:
                    dhw_demand = self.forecasts[b]['dhw_demand'][0]
                DHW_action = self.env.action_space[0].low[3 * b]

                # CASE 1: Solar generation is able to fulfill the demand
                diff_energy = self.forecasts[b]['solar_generation'][0] - (self.params['increase_scale_outage'] * cooling_elec_demand + 0.0 * dhw_demand + 0.0 * self.forecasts[b]['non_shiftable_load'][0])
                if diff_energy > 0.0:
                    # Use diff_energy to charge the battery (only if it is below max outage soc)
                    if self.cur_battery_soc[b] < self.params['max_soc_outage']:  # Use the generated energy to charge battery
                        ELE_action = self.compute_battery_action_given_demand(b, -diff_energy)  # Charge
                    else: # We don't need to charge battery so use the energy to fulfill the DHW demand if needed
                        ELE_action = 0.0
                else:
                    diff_energy2 = diff_energy + max_elec_in_battery
                    if diff_energy2 > 0.0:
                        # CASE 2: Solar generation + available in battery is able to fulfill the demand
                        # Give the needed (demand-generation) to the building from the battery
                        ELE_action = self.compute_battery_action_given_demand(b, -diff_energy)  # Discharge

                    else:
                        # CASE 3: Solar generation + available in battery is not able to fulfill the demand
                        ELE_action = - max_soc_reduction

                DHW_action = np.clip(DHW_action, self.env.action_space[0].low[3 * b], self.env.action_space[0].high[3 * b])
                self.predicted_dhw_device_demand[b] = self.compute_pred_dhw_device_consumption(b, DHW_action)

                ELE_action = np.clip(ELE_action, self.env.action_space[0].low[3 * b + 1], self.env.action_space[0].high[3 * b + 1])
                self.predicted_battery_demand[b], _ = self.compute_pred_battery_consumption(b, ELE_action)
                TMP_action = np.clip(TMP_action, self.env.action_space[0].low[3 * b + 2], self.env.action_space[0].high[3 * b + 2])
                self.predicted_cooling_demand[b], _ = self.cooling_device_controller[b].compute_pred_cooling_consumption(TMP_action, predicted_outdoor_temp)


            else:  # No outage, normal operation
                if self.outage_details[b]['outage_previously']:
                    self.outage_details[b]['time_since_last_outage'] += 1  # Add one hour

                if self.outage_details[b]['outage_duration'] > 0:  # Just after the outage
                    self.outage_details[b]['time_since_last_outage'] = 0
                    self.outage_details[b]['outage_duration'] = 0

                # COOLING ACTION - PID - XGBOOST
                TMP_action = self.cooling_device_controller[b].find_best_action(observations, predicted_outdoor_temp, last_step_cooling_demand, self.outage_details[b], expected_available_elec=np.inf)
                TMP_action = np.clip(TMP_action, self.env.action_space[0].low[3 * b + 2], self.env.action_space[0].high[3 * b + 2])
                self.predicted_cooling_demand[b], _ = self.cooling_device_controller[b].compute_pred_cooling_consumption(TMP_action, predicted_outdoor_temp)

                # DHW ACTION
                # will refine it later
                building_dhw_demand = self.forecasts[b]['dhw_demand'][0]
                avg_dhw_demand = 0
                for h in range(24):
                    h_avg = self.forecast_agent.hourly_expected_values['dhw_demand'][b][h]
                    if len(h_avg) > 0:
                        avg_dhw_demand += np.mean(h_avg)
                avg_dhw_demand /= 24
                heat_water = building_dhw_demand < avg_dhw_demand and self.cur_dhw_soc[b] < 0.9

                if heat_water:
                    DHW_action = self.env.action_space[0].high[3 * b + 2] * 0.20
                else:
                    DHW_action = -0.83

                DHW_action = np.clip(DHW_action, self.env.action_space[0].low[3 * b], self.env.action_space[0].high[3 * b])
                self.predicted_dhw_device_demand[b] = self.compute_pred_dhw_device_consumption(b, DHW_action)

                # BATTERY ACTION: leave it as 0, will compute later.
                ELE_action = 0.0

            action_proposals.extend([DHW_action, ELE_action, TMP_action])

        return action_proposals

    def compute_elec_consumption_values(self):
        # Calculate expected consumption
        predicted_consumption_per_b = np.zeros(self.n_buildings)
        for b in range(self.n_buildings):
            # The consumed electricity: Non-shiftable load + Cooling demand + DHW demand + Battery demand - PV generation
            predicted_consumption_per_b[b] = (self.forecasts[b]['non_shiftable_load'][0] + self.predicted_cooling_demand[b] + self.predicted_dhw_device_demand[b] + self.predicted_battery_demand[b] - self.forecasts[b]['solar_generation'][0])

            if len(self.elec_consumption_prediction_history[b]) > self.seen_steps:
                self.elec_consumption_prediction_history[b][-1] = predicted_consumption_per_b[b]
            else:
                self.elec_consumption_prediction_history[b].append(predicted_consumption_per_b[b])

    def get_future_cooling_demands(self, obs, action_proposals):
        future_cooling_demands = np.zeros((self.n_buildings, self.params['tau']))
        future_outdoor_temps = self.forecasts['outdoor_temp']
        for b in range(self.n_buildings):
            aux_pid_controller = self.cooling_device_controller[b].aux_pid_controller
            aux_pid_controller.integral = self.cooling_device_controller[b].pid_controller.integral
            aux_pid_controller.prev_error = self.cooling_device_controller[b].pid_controller.prev_error
            indoor_temp = obs[self.observation_names_b.index(f'indoor_dry_bulb_temperature_{b}')]
            setpoint = obs[self.observation_names_b.index(f'indoor_dry_bulb_temperature_set_point_{b}')]
            for t in range(self.params['tau']):
                pred_out_temp = future_outdoor_temps[t]

                action = aux_pid_controller.get_actions(b, indoor_temp, setpoint, 0.0, False, pred_out_temp)
                action = np.clip(action, self.env.action_space[0].low[3 * b + 2], self.env.action_space[0].high[3 * b + 2])
                if t == 0:  # take the real action taken
                    action = action_proposals[3 * b + 2]

                future_cooling_demands[b, t] = self.cooling_device_controller[b].compute_pred_cooling_consumption(action, pred_out_temp)[0]
                indoor_temp = setpoint  # Let's assume we reach the setpoint and the setpoint keeps constant

        return future_cooling_demands

    def get_future_dhw_demands(self, observations):
        hour = int(observations[self.observation_names_b.index('hour')])
        used_hour = hour-1 if hour != 0 else 23
        schedules = self.cur_dhw_heat_schedules
        future_dhw_demands = np.zeros((self.n_buildings, self.params['tau']))

        for t in range(self.params['tau']):
            h = used_hour + t if used_hour + t < 24 else used_hour + t - 24
            for b in range(self.n_buildings):
                future_dhw_demands[b, t] = schedules[b, h] * self.dhw_storage_capacity[b]

        return future_dhw_demands

    def get_consumption_forecast(self, hour):
        consumption_forecasts_per_b = np.zeros((self.n_buildings, self.params['tau']+1))
        for b in range(self.n_buildings):
            # first index is for the current consumption
            consumption_forecasts_per_b[b, 0] = self.elec_consumption_history[b][-1]
            for t in range(self.params['tau']):
                if t == 0:
                    cooling_demand = self.predicted_cooling_demand[b]
                    dhw_demand = self.predicted_dhw_device_demand[b]
                else:
                    used_hour = hour + t if hour + t < 24 else hour + t - 24

                    cooling_demand = self.future_cooling_demands[b, t]
                    expected_dhw_demand = self.forecast_agent.hourly_expected_values['dhw_demand'][b][used_hour]
                    dhw_demand = np.mean(expected_dhw_demand) if len(expected_dhw_demand) > 0 else self.predicted_dhw_device_demand[b]

                    expected_cooling_demand = self.forecast_agent.hourly_expected_values['cooling_demand'][b][used_hour]
                    if len(expected_cooling_demand) > 0:
                        cooling_demand += np.mean(expected_cooling_demand)
                        cooling_demand /= 2


                consumption_forecasts_per_b[b, t+1] = self.forecasts[b]['non_shiftable_load'][t] + cooling_demand + dhw_demand - self.forecasts[b]['solar_generation'][t]
                #consumption_forecasts_per_b[b, t+1] = (self.forecasts[b]['non_shiftable_load'][t] + self.predicted_dhw_device_demand[b]
                #                                       + self.predicted_cooling_demand[b] + self.predicted_battery_demand[b] - self.forecasts[b]['solar_generation'][t])

        return consumption_forecasts_per_b

    def refine_actions_with_battery_controller(self, hour, action_proposals):
        """
        Given the action proposals from the buildings
        Consumption forecast: it includes the last real consumption to consider ramping
        """
        normal_building_idx = [b for b in range(self.n_buildings) if not self.outage_details[b]['outage_flag']]

        consumption_forecast = self.get_consumption_forecast(hour)

        final_actions = action_proposals.copy()
        for b in normal_building_idx:
            #  state: [Avg balance, Cur SOC, Cur balance, t+1 balance, t+2 balance, ..., t+tau balance]
            avg_balance = np.array(self.elec_consumption_history[b]).mean()
            std_balance = np.array(self.elec_consumption_history[b]).std()

            # TREE-SEARCH FOR BATTERY ACTION
            state = np.array([avg_balance, self.cur_battery_soc[b], *consumption_forecast[b, :]])
            action, cost = self.battery_controller[b].search(state, hour=hour)
            final_actions[3 * b + 1] = action[0]

            self.predicted_battery_demand[b], _ = self.compute_pred_battery_consumption(b, action[0])

            # Check if after the battery action we are still consuming out of the bounds
            next_step_total_consumption = consumption_forecast[b, 1] + self.predicted_battery_demand[b]

            # CONSUMING MORE THAN HIGH BOUND
            if next_step_total_consumption > avg_balance + self.params['B_high'] * std_balance:
                # We are consuming more than expected even after battery action
                consumption_to_be_reduced = next_step_total_consumption - avg_balance #- self.params['B_high'] * std_balance

                # AUX STEP 1: reduce water heating
                DHW_action = final_actions[3 * b]
                if (DHW_action > 0.0) and (consumption_to_be_reduced > 0):  # In case we are heating water
                    avoidable_elec = self.predicted_dhw_device_demand[b]
                    # reduce it entirely
                    consumption_to_be_reduced -= avoidable_elec
                    # negative in case there is any demand
                    final_actions[3 * b] = self.env.action_space[0].low[3 * b]
                    self.predicted_dhw_device_demand[b] = self.compute_pred_dhw_device_consumption(b, final_actions[3 * b])

                # AUX STEP 2: reduce cooling demand
                TMP_action = final_actions[3 * b + 2]
                avoidable_elec = self.params['TMP_max_reduction_percent'] * self.predicted_cooling_demand[b]
                if (consumption_to_be_reduced > 0) and (avoidable_elec > consumption_to_be_reduced):
                    # reduce the cooling demand just a bit (less than the max amount)
                    # consumption_to_be_reduced = 0
                    final_actions[3 * b + 2] = TMP_action * (1 - self.params['TMP_max_reduction_percent'])
                else:
                    # reduce it entirely
                    consumption_to_be_reduced -= avoidable_elec
                    final_actions[3 * b + 2] = TMP_action * (1 - self.params['TMP_max_reduction_percent'])
                pred_out_temp = self.forecasts['outdoor_temp'][0]
                self.predicted_cooling_demand[b], _ = self.cooling_device_controller[b].compute_pred_cooling_consumption(final_actions[3 * b + 2], pred_out_temp)

            # CONSUMING LESS THAN LOW BOUND
            elif next_step_total_consumption < avg_balance - self.params['B_low'] * std_balance:
                # consuming less than expected even after battery action
                consumption_to_be_increased = avg_balance - next_step_total_consumption - self.params['B_low'] * std_balance
                assert consumption_to_be_increased > 0, "This should not happen"

                # AUX STEP 1: increase water heating
                DHW_action = final_actions[3 * b]
                building_dhw_demand = self.forecasts[b]['dhw_demand'][0]
                # Get avg dhw consumption
                avg_dhw_demand = 0
                for h in range(24):
                    h_avg = self.forecast_agent.hourly_expected_values['dhw_demand'][b][h]
                    if len(h_avg) > 0:
                        avg_dhw_demand += np.mean(h_avg)
                avg_dhw_demand /= 24

                if (building_dhw_demand < avg_dhw_demand) and (self.cur_dhw_soc[b] < 0.95) and (consumption_to_be_increased > 0):  # In case we are not heating water
                    if DHW_action < 0.0:
                        # provide hot water
                        left_to_charge = (1 - self.cur_dhw_soc[b]) * self.dhw_storage_capacity[b]
                        if left_to_charge > consumption_to_be_increased: # We can increase the water heating demand just a bit (not entirely)
                            consumption_to_be_increased = 0
                            final_actions[3 * b] = self.dhw_device_efficiency[b] * (left_to_charge - consumption_to_be_increased) / self.dhw_storage_capacity[b]

                        else:
                            # increase it entirely
                            consumption_to_be_increased -= left_to_charge
                            final_actions[3 * b] = self.env.action_space[0].high[3 * b]

                    self.predicted_dhw_device_demand[b] = self.compute_pred_dhw_device_consumption(b, final_actions[3 * b])

        # clip all the actions
        for b in range(self.n_buildings):
            DHW_action = final_actions[3 * b]
            ELE_action = final_actions[3 * b + 1]
            TMP_action = final_actions[3 * b + 2]
            final_actions[3 * b] = np.clip(DHW_action, self.env.action_space[0].low[3 * b], self.env.action_space[0].high[3 * b])
            final_actions[3 * b + 1] = np.clip(ELE_action, self.env.action_space[0].low[3 * b + 1], self.env.action_space[0].high[3 * b + 1])
            final_actions[3 * b + 2] = np.clip(TMP_action, self.env.action_space[0].low[3 * b + 2], self.env.action_space[0].high[3 * b + 2])
        return final_actions

    def compute_pred_battery_consumption(self, b, ELE_action):
        ELE_action = np.clip(ELE_action, self.env.action_space[0].low[3 * b + 1], self.env.action_space[0].high[3 * b + 1])

        energy = ELE_action * self.battery_capacities[b]

        # Calculate the capacity power curve
        max_input_power = self.calculate_battery_max_input_power(b, self.cur_battery_soc[b])

        # Now calculate the efficiency with efficiency curve
        efficiency = self.calculate_battery_efficiency(b, energy)

        cur_energy = self.cur_battery_soc[b] * self.battery_capacities[b]

        if energy >= 0:  # Positive action - Charging
            energy = min(max_input_power, self.battery_nominal_powers[b], energy)
            energy_final = min(cur_energy + energy * efficiency, self.battery_capacities[b])

            electricity_consumption = (energy_final - cur_energy) / efficiency
        else:  # Negative action - Discharging
            #soc_limit_wrt_dod = 1.0 - self.min_battery_soc[b]   # (depth of discharge)
            soc_difference = self.cur_battery_soc[b] - self.min_battery_soc[b]
            energy_limit_wrt_dod = -1 * max(soc_difference * self.battery_capacities[b] * efficiency, 0.0)
            energy = max(-max_input_power, energy_limit_wrt_dod, energy)
            energy_final = max(0.0, cur_energy + energy / efficiency)

            electricity_consumption = (energy_final - cur_energy) * efficiency

        return electricity_consumption, efficiency

    def calculate_battery_efficiency(self, b, energy):
        energy_normalized = np.abs(energy) / self.battery_nominal_powers[b]
        idx = max(0, np.argmax(energy_normalized <= self.battery_efficiency_curves[b][0]) - 1)
        efficiency = (self.battery_efficiency_curves[b][1][idx] + (energy_normalized - self.battery_efficiency_curves[b][0][idx])
                      * (self.battery_efficiency_curves[b][1][idx + 1] - self.battery_efficiency_curves[b][1][idx])
                      / (self.battery_efficiency_curves[b][0][idx + 1] - self.battery_efficiency_curves[b][0][idx]))
        return efficiency

    def calculate_battery_max_input_power(self, b, soc):
        idx = max(0, np.argmax(soc <= self.battery_capacity_power_curves[b][0]) - 1)
        max_input_power = (self.battery_nominal_powers[b]
                           * (self.battery_capacity_power_curves[b][1][idx] + (self.battery_capacity_power_curves[b][1][idx + 1] - self.battery_capacity_power_curves[b][1][idx])
                              * (soc - self.battery_capacity_power_curves[b][0][idx])
                              / (self.battery_capacity_power_curves[b][0][idx + 1] - self.battery_capacity_power_curves[b][0][idx])))
        return max_input_power

    def compute_battery_action_given_demand(self, b, ELE_demand):
        efficiency = self.calculate_battery_efficiency(b, ELE_demand)
        input_energy = ELE_demand * efficiency
        needed_action = input_energy / self.battery_capacities[b]
        initial_guess = -1 * needed_action
        final_action = initial_guess

        return final_action

    def compute_pred_dhw_device_consumption(self, b, DHW_action):
        DHW_action = np.clip(DHW_action, self.env.action_space[0].low[3 * b], self.env.action_space[0].high[3 * b])
        energy = DHW_action * self.dhw_storage_capacity[b]  # energy willing to be used
        cur_energy = self.cur_dhw_soc[b] * self.dhw_storage_capacity[b]  # max energy available

        demand = self.forecasts[b]['dhw_demand'][0]
        if demand < 0.15:
            demand = 0.0

        if energy > 0.0:  # Positive action - Heating water
            downward_electrical_flexibility = np.inf
            max_output = min(downward_electrical_flexibility, self.dhw_device_nominal_powers[b]) * self.dhw_device_efficiency[b]
            energy = min(max_output, energy)  # Clip to max_output
            consumption = energy + demand
        else: # Negative action - Using hot water
            used_energy = min(abs(energy), cur_energy)

            if used_energy > demand:
                # We have enough hot water.
                consumption = 0.0
            else:
                # We will use the hot water available and use elec to heat the rest
                consumption = demand - used_energy
        electricity_consumption = consumption / self.dhw_device_efficiency[b]
        return electricity_consumption
