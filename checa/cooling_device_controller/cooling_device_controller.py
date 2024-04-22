import numpy as np
from checa.cooling_device_controller.pid_controller import PIDController
from scipy import stats


class CoolingDeviceController:
    def __init__(self, building_metadata, b, obs_names_b):
        self.building_metadata = building_metadata
        self.observation_names_b = obs_names_b
        self.b = b
        self.seen_steps = 0

        self.update_freq = 24
        self.n = 2.0  # exponent in the error to update scores

        self.cooling_nominal_powers = self.building_metadata['cooling_device']['nominal_power']
        self.cooling_device_efficiency = self.building_metadata['cooling_device']['efficiency']
        self.cooling_cop = 0.0

        # PID
        # Initialize PID controller
        self.use_pid = True

        # Controller for all three buildings
        self.pid_controller = PIDController()
        self.pid_controller.Kp = -0.288074675781558  # -0.0183231
        self.pid_controller.Ki = -2.489908316536426  # -4.298114009977
        self.pid_controller.Kd = 0.009479951268930725  # 0.0 to be a PI controller
        self.pid_controller.dt = 0.1  # 0.1
        self.pid_controller.OutTempScaler = 0.09629375171771082
        self.pid_controller.Kp_outage = -0.27351617926369537

        self.aux_pid_controller = PIDController()
        self.aux_pid_controller.Kp = -0.288074675781558
        self.aux_pid_controller.Ki = -2.489908316536426
        self.aux_pid_controller.Kd = 0.009479951268930725
        self.aux_pid_controller.dt = 0.1
        self.aux_pid_controller.OutTempScaler = 0.09629375171771082
        self.aux_pid_controller.Kp_outage = -0.27351617926369537

        """
        {"target": Dont know, "params": {"Kp_o": -0.2859441319844675, "OutTempScaler": 0.1024468677938111}, Average score: Average score: 0.500, Resilience: 0.354, Unserved: 0.442

        {"target": -0.505676805973053, "params": {"Kp_o": -0.27351617926369537, "OutTempScaler": 0.09629375171771082}, Average score: 0.499, Resilience: 0.345, Unserved: 0.442
        {"target": -0.5059506297111511, "params": {"Kp_o": -0.2811766922821901, "OutTempScaler": 0.08356605234881731},
        {"target": -0.5045549869537354, "params": {"Kp_o": -0.2692889293697175, "OutTempScaler": 0.08117311805607505}, Average score: 0.499, Resilience: 0.349, Unserved: 0.440
        {"target": -0.5058454275131226, "params": {"Kp_o": -0.2788985853672359, "OutTempScaler": 0.087780233592517}, 
        {"target": -0.5039359331130981, "params": {"Kp_o": -0.2675317659029748, "OutTempScaler": 0.03640156612555039}, Average score: 0.499, Resilience: 0.351, Unserved: 0.436
        {"target": -0.5039920210838318, "params": {"Kp_o": -0.26547226577692423, "OutTempScaler": 0.03181194000391644},
        {"target": -0.5061987042427063, "params": {"Kp_o": -0.2860233588478051, "OutTempScaler": 0.1}, Average score: 0.501, Resilience: 0.359, Unserved: 0.442
        {"target": -0.5043141841888428, "params": {"Kp_o": -0.2574072807620056, "OutTempScaler": 0.032388445279158365}, Average score: 0.499, Resilience: 0.355, Unserved: 0.435
        
        
        AVG SCORE MULTIPLE SEEDS
        {"target": -0.5067414045333862, "params": {"Kp_o": -0.247013260264578, "OutTempScaler": 0.03100663702805051}, "datetime": {"datetime": "2023-11-12 20:23:34", "elapsed": 12329.972161, "delta": 583.082663}}
        {"target": -0.506675660610199, "params": {"Kp_o": -0.2520716329315423, "OutTempScaler": 0.02476989675830077}, "datetime": {"datetime": "2023-11-12 17:01:46", "elapsed": 0.0, "delta": 0.0}}

        OUTAGE
        {"target": -0.7852861991014534, "params": {"Kp_o": -0.27718231017358447, "OutTempScaler": 0.10122917387682927}, "datetime": {"datetime": "2023-11-12 20:38:48", "elapsed": 12542.656823, "delta": 582.264053}}
        {"target": -0.7863135214158126, "params": {"Kp_o": -0.2661926894272662, "OutTempScaler": 0.040887324044468536}, "datetime": {"datetime": "2023-11-12 21:00:47", "elapsed": 13885.767592, "delta": 573.705677}}
        {"target": -0.7842492774382114, "params": {"Kp_o": -0.2589563545239867, "OutTempScaler": 0.03621714354693686}, "datetime": {"datetime": "2023-11-12 19:08:24", "elapsed": 6866.007594, "delta": 862.97067}}
        {"target": -0.7805302212058559, "params": {"Kp_o": -0.2520716329315423, "OutTempScaler": 0.02476989675830077}, "datetime": {"datetime": "2023-11-12 17:09:46", "elapsed": 0.0, "delta": 0.0}}
        {"target": -0.7813702335728071, "params": {"Kp_o": -0.2564384096876519, "OutTempScaler": 0.025462126196686148}, "datetime": {"datetime": "2023-11-12 21:07:34", "elapsed": 14266.482332, "delta": 581.023997}}

        RESILIENCE
        {"target": -0.35921516754850086, "params": {"Kp_o": -0.3087273271121387, "OutTempScaler": 0.10495128632644757}, "datetime": {"datetime": "2023-11-12 18:31:28", "elapsed": 4855.576308, "delta": 603.819682}}
        {"target": -0.3574514991181657, "params": {"Kp_o": -0.3999607691153279, "OutTempScaler": 0.09733470317930655}, "datetime": {"datetime": "2023-11-12 20:44:13", "elapsed": 12809.625391, "delta": 591.020145}}
        """

        self.setpoint_history = []
        self.last_setpoint = None
        self.used_setpoints = []

        self.proposed_demand = None

        self.saturation_perc_limit = 0.5493411704104555 #0.2
        self.added_temp_to_setpoint = 0.1152247939657649 #0.0

    def step(self):
        self.seen_steps += 1

    def find_best_action(self, obs, pred_out_temp, last_step_demand, outage_details,
                         expected_available_elec):  # TODO use last step action to update coefficients
        """
        Given the current state, find the action that leads to the setpoint.
        state: [direct_solar_irradiance, diffuse_solar_irradiance, outdoor_temp, indoor_temp, action (cooling demand - not real action)]
        """

        indoor_temp = obs[self.observation_names_b.index(f'indoor_dry_bulb_temperature_{self.b}')]
        temp_setpoint = obs[self.observation_names_b.index(f'indoor_dry_bulb_temperature_set_point_{self.b}')]
        self.setpoint_history.append(temp_setpoint)

        if self.proposed_demand is not None:
            if self.proposed_demand > 0:
                saturated_perc = (abs(last_step_demand - self.proposed_demand) / self.proposed_demand)
            else:
                saturated_perc = 0.0
            if saturated_perc < self.saturation_perc_limit:
                saturated_perc = 0.0
        else:
            saturated_perc = 0.0
        outage_details['saturation_perc'] = saturated_perc
        if not outage_details['saturated_previously'] and saturated_perc >= self.saturation_perc_limit:
            outage_details['saturated_previously'] = True

        use_mode_setpoint = False
        if use_mode_setpoint:
            # set the mode of the last 4 hours
            mode, count = stats.mode(self.setpoint_history[-4:])
            if count > 1:
                used_setpoint = mode
            else:
                used_setpoint = temp_setpoint
        else:
            used_setpoint = temp_setpoint
        self.used_setpoints.append(used_setpoint)

        # CASE 1: No outage: use PID controller as normal
        reduced_setpoint = used_setpoint + self.added_temp_to_setpoint  # FOr outage

        if not outage_details['outage_flag']:
            # If after outage, we are higher from setpoint, reset integral and prev_error
            after_outage = outage_details['outage_previously'] and outage_details['time_since_last_outage'] == 0
            if after_outage and outage_details['saturated_previously']:
                self.pid_controller.integral = 0
                self.pid_controller.updated_integral = 0
                self.pid_controller.prev_error = 0
                self.pid_controller.updated_prev_error = 0
                outage_details['saturated_previously'] = False

            demand_action = self.pid_controller.get_actions(self.b, indoor_temp, reduced_setpoint, 0.0, False, pred_out_temp)

        # CASE 2: Outage but no saturation
        elif saturated_perc < self.saturation_perc_limit:
            # check if batteries will give enough
            demand_action = self.pid_controller.get_actions(self.b, indoor_temp, reduced_setpoint, 0.0, False,
                                                            pred_out_temp)
            if demand_action < expected_available_elec:
                pass  # enough battery and gen: use as normal
            else:
                # Then we will surely enter in saturation if outage continues
                diff_temp = indoor_temp - used_setpoint
                if diff_temp > 1.5:
                    demand_action = self.pid_controller.get_actions(self.b, indoor_temp, reduced_setpoint, 0.0, True, pred_out_temp)
                else:
                    demand_action = self.pid_controller.get_actions(self.b, indoor_temp, reduced_setpoint, 0.0, False, pred_out_temp)

        # CASE 3: Outage and saturation: use PID controller with reduced action, see how the diff temp is
        else:
            # check if next step battery will be enough as a not outage flag
            # demand_action = self.pid_controller.get_actions(self.b, indoor_temp, used_setpoint, saturated_perc, False, pred_out_temp)

            # if demand_action < expected_available_elec:
            #    pass # enough battery and gen: use as normal even we had saturation

            # else: # Not enough battery and gen. In this case, maybe we are getting out from outage in the next step... so be careful
            diff_temp = indoor_temp - used_setpoint
            if diff_temp>1.5:
                demand_action = self.pid_controller.get_actions(self.b, indoor_temp, reduced_setpoint, saturated_perc, True, pred_out_temp)
            else:
                demand_action = self.pid_controller.get_actions(self.b, indoor_temp, reduced_setpoint, saturated_perc, False, pred_out_temp)

        if demand_action > expected_available_elec:
            demand_action = expected_available_elec

        action = self.compute_cooling_action_given_elec_demand(demand_action)

        action = np.clip(action, 0.0, 1.0)

        self.proposed_demand = self.compute_pred_cooling_consumption(action, pred_out_temp)[1]

        self.step()

        self.last_setpoint = temp_setpoint

        return action

    def compute_pred_cooling_consumption(self, TMP_action, outdoor_temp):
        self.get_cop(outdoor_temp)

        electric_power = TMP_action * self.cooling_nominal_powers
        # we are assuming available nominal power == all nominal power
        demand = min(electric_power, self.cooling_nominal_powers) * self.cooling_cop

        # Calculate device output energy and electricity consumption
        downward_electrical_flexibility = np.inf
        max_input_power = min(downward_electrical_flexibility, self.cooling_nominal_powers) * self.cooling_cop

        device_output = min(max_input_power, demand)

        electricity_consumption = device_output / self.cooling_cop

        return electricity_consumption, demand

    def get_cop(self, outdoor_temp):
        t_target_cooling = self.building_metadata['cooling_device']['target_cooling_temperature']
        cop = (t_target_cooling + 273.15) * self.cooling_device_efficiency / (outdoor_temp - t_target_cooling)
        if cop < 0 or cop > 20:
            cop = 20
        self.cooling_cop = cop

    def compute_cooling_action_given_elec_demand(self, cooling_elec_demand):
        return cooling_elec_demand / self.cooling_nominal_powers
