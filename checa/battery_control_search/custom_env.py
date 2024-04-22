import numpy as np
import pandas as pd
import torch


class DataGenerator:
    def __init__(self):
        building_1_df = pd.read_csv('../../data/schemas/warm_up/Building_1.csv')
        building_1_df['building'] = 0
        building_1_df['battery_capacity'] = 4.0  # kWh (cannot be less than 20% of max capacity - 800Wh)
        building_1_df['solar_generation'] = building_1_df['Solar Generation (W/kW)'] * 2.4 / 1000
        building_2_df = pd.read_csv('../../data/schemas/warm_up/Building_2.csv')
        building_2_df['building'] = 1
        building_2_df['battery_capacity'] = 4.0  # kWh (cannot be less than 20% of max capacity - 660Wh)
        building_2_df['solar_generation'] = building_1_df['Solar Generation (W/kW)'] * 1.2 / 1000
        building_3_df = pd.read_csv('../../data/schemas/warm_up/Building_3.csv')
        building_3_df['building'] = 2
        building_2_df['battery_capacity'] = 3.3  # kWh (cannot be less than 20% of max capacity - 660Wh)
        building_3_df['solar_generation'] = building_1_df['Solar Generation (W/kW)'] * 2.4 / 1000

        self.building_df = pd.concat([building_1_df, building_2_df, building_3_df])
        self.solar_gen_avg_per_hour = self.building_df.groupby('Hour').mean()['solar_generation'].values
        self.solar_gen_std_per_hour = self.building_df.groupby('Hour').std()['solar_generation'].values
        self.non_shiftable_load_avg_per_hour = self.building_df.groupby('Hour').mean()['Equipment Electric Power (kWh)'].values
        self.non_shiftable_load_std_per_hour = self.building_df.groupby('Hour').std()['Equipment Electric Power (kWh)'].values
        self.dhw_demand_avg_per_hour = self.building_df.groupby('Hour').mean()['DHW Heating (kWh)'].values
        self.dhw_demand_std_per_hour = self.building_df.groupby('Hour').std()['DHW Heating (kWh)'].values
        self.cooling_demand_avg_per_hour = self.building_df.groupby('Hour').mean()['Cooling Load (kWh)'].values
        self.cooling_demand_std_per_hour = self.building_df.groupby('Hour').std()['Cooling Load (kWh)'].values


    def solar_generation(self, hour, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(self.solar_gen_avg_per_hour[hour], self.solar_gen_std_per_hour[hour])

    def non_shiftable_load(self, hour, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(self.non_shiftable_load_avg_per_hour[hour], self.non_shiftable_load_std_per_hour[hour])

    def dhw_demand(self, hour, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(self.dhw_demand_avg_per_hour[hour], self.dhw_demand_std_per_hour[hour])

    def cooling_demand(self, hour, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(self.cooling_demand_avg_per_hour[hour], self.cooling_demand_std_per_hour[hour])


class Env:
    def __init__(self, n_buildings=1, total_time_steps=10, tau=3, seed=None):
        """
        Env for testing GNN controller
        """
        self.generator = DataGenerator()
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.n_buildings = n_buildings
        self.tau = tau  # forecasting horizon to be considered
        self.state_dim = 3 + tau
        self.total_time_steps = total_time_steps

        self.time_step = 0
        self.hour = np.random.randint(0, 24)
        self.battery_capacities = np.array([np.random.uniform(3.0, 4.0) for _ in range(self.n_buildings)])
        self.battery_soc = np.array([np.random.uniform(0.0, 1.0) for _ in range(self.n_buildings)])

        self.solar_generation = np.zeros((self.total_time_steps+self.tau, self.n_buildings))
        self.non_shiftable_load = np.zeros((self.total_time_steps+self.tau, self.n_buildings))
        self.dhw_demand = np.zeros((self.total_time_steps+self.tau, self.n_buildings))
        self.cooling_demand = np.zeros((self.total_time_steps+self.tau, self.n_buildings))

        # Electricity balance
        self.electricity_balance = np.zeros((self.total_time_steps, self.n_buildings))
        self.avg_electricity_balance = np.zeros(self.n_buildings)

        # For computing Reward: baseline with no battery usage
        self.electricity_balance_baseline = np.zeros((self.total_time_steps, self.n_buildings))
        self.baseline_rewards = (0, 0)

        # Predictions of the balance
        self.electricity_balance_predictions = np.zeros((self.total_time_steps+self.tau, self.n_buildings))

        # State
        self.state = np.zeros((self.n_buildings, self.state_dim))

        # Misc
        self.plot = False
        self.print = False

    def reset(self):
        self.time_step = 0
        self.hour = np.random.randint(0, 24)
        self.battery_soc = np.array([np.random.uniform(0.2, 1.0) for _ in range(self.n_buildings)])
        self.electricity_balance = np.zeros((self.total_time_steps, self.n_buildings))
        self.electricity_balance_predictions = np.zeros((self.total_time_steps+self.tau, self.n_buildings))

        # Generate demand and generation data for self.total_time_steps
        for b in range(self.n_buildings):
            hour = self.hour
            for t in range(self.total_time_steps):
                self.solar_generation[t, b] = self.generator.solar_generation(hour, self.seed)
                self.non_shiftable_load[t, b] = self.generator.non_shiftable_load(hour, self.seed)
                self.dhw_demand[t, b] = self.generator.dhw_demand(hour, self.seed)
                self.cooling_demand[t, b] = self.generator.cooling_demand(hour, self.seed)
                self.electricity_balance_baseline[t, b] = self.non_shiftable_load[t, b] + self.dhw_demand[t, b] + self.cooling_demand[t, b] - self.solar_generation[t, b]
                # TODO: not use real data as predictions, add noise
                self.electricity_balance_predictions[t, b] = self.non_shiftable_load[t, b] + self.dhw_demand[t, b] + self.cooling_demand[t, b] - self.solar_generation[t, b]
                hour = (hour + 1) % 24
            for t in range(self.total_time_steps, self.total_time_steps+self.tau):
                self.solar_generation[t, b] = self.generator.solar_generation(hour)
                self.non_shiftable_load[t, b] = self.generator.non_shiftable_load(hour)
                self.dhw_demand[t, b] = self.generator.dhw_demand(hour)
                self.cooling_demand[t, b] = self.generator.cooling_demand(hour)
                self.electricity_balance_predictions[t, b] = self.non_shiftable_load[t, b] + self.dhw_demand[t, b] + self.cooling_demand[t, b] - self.solar_generation[t, b]
                hour = (hour + 1) % 24

        # Update the first time step to be on the average of the consumptions

        self.avg_electricity_balance = np.mean(self.electricity_balance_baseline, axis=0)
        self.electricity_balance[0, :] = self.electricity_balance_baseline[0, :]

        # Compute initial state
        self.state = self.get_state()

        return self.state

    def step(self, actions, eval_step=False):
        """actions: list of length n_buildings, percentage of battery to charge/discharge [-1, 1]"""
        if eval_step or self.print:
            print(f"Time step: {self.time_step} Cur soc {self.battery_soc[0]:.3f}. Cur balance: {self.electricity_balance[self.time_step][0]:.3f} --> performing action: {actions[0]:.3f}")

        self.time_step += 1
        done = self.time_step + 1 == self.total_time_steps

        # Apply actions and update electricity balance
        for b in range(self.n_buildings):
            action = actions[b]
            max_action = 1.0 - self.battery_soc[b]
            min_action = 0.2 - self.battery_soc[b]
            #action = min_action + (max_action - min_action) * (action + 1) / 2  # rescale from [-1, 1] to [min_action, max_action]
            action = np.clip(action, min_action, max_action)  # clip between max and min action
            self.battery_soc[b] = self.battery_soc[b] + action

            # calculate electricity balance
            consumption = self.non_shiftable_load[self.time_step, b] + self.dhw_demand[self.time_step, b] + self.cooling_demand[self.time_step, b]
            generation = - self.solar_generation[self.time_step, b]
            battery_consumption = action * self.battery_capacities[b]
            balance = consumption + generation + battery_consumption
            self.electricity_balance[self.time_step, b] = balance


        self.state = self.get_state()
        if eval_step or self.print:
            print(f"New balance: {self.electricity_balance[self.time_step][0]:.3f}")
        reward = self.compute_intermediate_reward()
        if done:
            ramp_r, max_elec_r = self.compute_final_reward()
            if eval_step or self.print:
                print(f"Ramping reward: {ramp_r:.4f}. Max elec reward: {max_elec_r:.4f}. Total reward: {reward[0]:.4f}")
            if eval_step or self.plot:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('TkAgg')

                x_values = np.arange(self.total_time_steps)
                # x values are hour values, starting at self.hour
                x_values = (x_values + self.hour) % 24
                plt.figure()
                plt.plot(np.sum(self.electricity_balance, axis=1), label='Obtained balance')
                plt.plot(np.sum(self.electricity_balance_baseline, axis=1), label='Baseline balance')
                # horizontal line in avg
                plt.plot(np.ones(self.total_time_steps) * np.mean(self.avg_electricity_balance), label='Avg balance')
                # use x_values as x axis
                plt.xticks(np.arange(self.total_time_steps), x_values)
                plt.legend()
                plt.title(str(ramp_r))
                plt.xlabel('Time step')
                plt.ylabel('Electricity balance')
                plt.show()

        return self.state, reward, done

    def get_state(self):
        """
        States: shape (n_buildings, 4 + tau)
            - Avg electricity balance: float [-inf, inf] (building specific)
            - Cur battery soc: float [0, 1] (building specific)
            - Cur step electricity balance: float [0, inf] (building specific)
            - Next tau step's electricity balance prediction: list[float] [0, inf] (building specific)
        """
        state = np.zeros((self.n_buildings, self.state_dim))
        for b in range(self.n_buildings):
            # 1) Cur avg electricity balance
            state[b, 0] = self.avg_electricity_balance[b]
            # 2) Cur battery soc
            state[b, 1] = self.battery_soc[b]
            # 3) Last step electricity balance
            state[b, 2] = self.electricity_balance[self.time_step, b]
            # 4) Next tau step's electricity balance prediction
            for t in range(self.tau):
                state[b, 3+t] = self.electricity_balance_predictions[self.time_step+t+1, b]

        return state

    def compute_intermediate_reward(self):
        # Be sure to only consider the first self.total_time_steps
        cur_elec_balance = self.electricity_balance[self.time_step, :]
        cur_elec_balance_baseline = self.electricity_balance_baseline[self.time_step, :]

        past_ele_balance = self.electricity_balance[self.time_step-1, :]
        past_ele_balance_baseline = self.electricity_balance_baseline[self.time_step-1, :]


        # Compute ramping reward
        ramping_reward = - abs(cur_elec_balance - past_ele_balance)
        ramping_reward_baseline = - abs(cur_elec_balance_baseline - past_ele_balance_baseline)

        return ramping_reward - ramping_reward_baseline

    def compute_final_reward(self):
        """
        Compute final rewards
        elec_balance: np.array of shape (total_time_steps, n_buildings)
        Reward: Ramping reward + Max consumption reward
            Ramping reward: mean(- abs(next hour's balance - last hour's balance))
            Max consumption reward: - max(ele_consumption)
        """
        # Be sure to only consider the first self.total_time_steps
        elec_balance = self.electricity_balance[:self.total_time_steps, :]
        elec_balance_baseline = self.electricity_balance_baseline[:self.total_time_steps, :]

        # Get total balance: sum of all buildings
        total_balance = np.sum(elec_balance, axis=1)
        total_balance_baseline = np.sum(elec_balance_baseline, axis=1)

        # Compute ramping reward
        ramping_reward = 0
        ramping_reward_baseline = 0
        for t in range(self.total_time_steps-1):
            ramping_reward += - abs(total_balance[t+1] - total_balance[t])
            ramping_reward_baseline += - abs(total_balance_baseline[t+1] - total_balance_baseline[t])

        ramping_reward = (ramping_reward - ramping_reward_baseline) / (self.total_time_steps - 1)

        # Compute max consumption reward
        max_consumption_reward = - max(total_balance)
        max_consumption_reward_baseline = - max(total_balance_baseline)
        max_consumption_reward = max_consumption_reward - max_consumption_reward_baseline
        return ramping_reward, max_consumption_reward

