import numpy as np
from citylearn.citylearn import EvaluationCondition
from citylearn.cost_function import CostFunction

CONTROL_CONDITION = EvaluationCondition.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV
BASELINE_CONDITION = EvaluationCondition.WITHOUT_STORAGE_AND_PARTIAL_LOAD_BUT_WITH_PV

get_net_electricity_consumption = lambda x, c: getattr(x, f'net_electricity_consumption{c.value}')
get_net_electricity_consumption_cost = lambda x, c: getattr(x, f'net_electricity_consumption_cost{c.value}')
get_net_electricity_consumption_emission = lambda x, c: getattr(x, f'net_electricity_consumption_emission{c.value}')


# COMFORT REWARD
def custom_comfort_no_outage_reward(env, comfort_band, return_all=False):
    """
    :param env: CityLearn environment
    :param comfort_band: comfort band for comfort reward
    """
    data = []
    for b in env.buildings:
        discomfort_kwargs = {
            'indoor_dry_bulb_temperature': b.indoor_dry_bulb_temperature,
            'dry_bulb_temperature_set_point': b.indoor_dry_bulb_temperature_set_point,
            'band': comfort_band,
            'occupant_count': b.occupant_count,
        }

        occupant_count = np.array(b.occupant_count, dtype='float32')
        power_outage = np.array(b.power_outage_signal, dtype='float32')
        occupant_count[power_outage==1.0] = 0.0
        discomfort_kwargs['occupant_count'] = occupant_count
        unmet = CostFunction.discomfort(**discomfort_kwargs)[0]
        data.append(unmet[-1])

    if return_all:
        return -np.mean(data), np.array(data)
    else:
        return -np.mean(data)


# GRID REWARD
def custom_grid_reward(env):
    """
    :param env: CityLearn environment
    """
    ramping_average = CostFunction.ramping(get_net_electricity_consumption(env, CONTROL_CONDITION))[-1] / \
                      CostFunction.ramping(get_net_electricity_consumption(env, BASELINE_CONDITION))[-1]

    daily_one_minus_load_factor_average = (
            CostFunction.one_minus_load_factor(get_net_electricity_consumption(env, CONTROL_CONDITION), window=24)[-1]
            / CostFunction.one_minus_load_factor(get_net_electricity_consumption(env, BASELINE_CONDITION), window=24)[-1])

    daily_peak_average = CostFunction.peak(get_net_electricity_consumption(env, CONTROL_CONDITION), window=24)[-1] / \
                         CostFunction.peak(get_net_electricity_consumption(env, BASELINE_CONDITION), window=24)[-1]

    annual_peak_average = CostFunction.peak(get_net_electricity_consumption(env, CONTROL_CONDITION), window=8760)[-1] / \
                          CostFunction.peak(get_net_electricity_consumption(env, BASELINE_CONDITION), window=8760)[-1]

    reward = (ramping_average + daily_one_minus_load_factor_average + daily_peak_average + annual_peak_average) / 4
    return -reward


def custom_efficiency_reward(env):
    """
    :param env: CityLearn environment
    """
    efficiency_reward = []
    for b in env.buildings:
        cooling_device_consumption = np.sum(b.cooling_device.electricity_consumption) / env.time_steps
        efficiency_reward.append(cooling_device_consumption)

    efficiency_reward = np.mean(efficiency_reward)
    return -efficiency_reward


# COMFORT AND EFFICIENCY REWARD
def custom_comfort_and_efficiency_reward(env, comfort_grid_tradeoff, comfort_band):
    """
    :param env: CityLearn environment
    :param comfort_grid_tradeoff: 0.0 means only grid, 1.0 means only comfort
    :param comfort_band: comfort band for comfort reward
    """
    comfort_reward = -custom_comfort_no_outage_reward(env, comfort_band)
    efficiency_reward = -custom_efficiency_reward(env)
    #print(f'efficiency_reward: {efficiency_reward}. comfort_reward: {comfort_reward}')
    reward = comfort_grid_tradeoff * comfort_reward + (1 - comfort_grid_tradeoff) * efficiency_reward
    return -reward


# OUTAGE REWARD
def custom_outage_reward(env, comfort_band, return_all=False):
    """
    :param env: CityLearn environment
    :param comfort_band: comfort band for comfort reward
    :param return_all: return all values for debugging
    """
    resilience = []
    unmet_demand = []
    for b in env.buildings:
        discomfort_kwargs = {
            'indoor_dry_bulb_temperature': b.indoor_dry_bulb_temperature,
            'dry_bulb_temperature_set_point': b.indoor_dry_bulb_temperature_set_point,
            'band': comfort_band,
            'occupant_count': b.occupant_count,
        }
        expected_energy = b.cooling_demand + b.heating_demand + b.dhw_demand + b.non_shiftable_load
        served_energy = b.energy_from_cooling_device + b.energy_from_cooling_storage \
                        + b.energy_from_heating_device + b.energy_from_heating_storage \
                        + b.energy_from_dhw_device + b.energy_from_dhw_storage \
                        + b.energy_to_non_shiftable_load

        resilience.append(CostFunction.one_minus_thermal_resilience(power_outage=b.power_outage_signal, **discomfort_kwargs)[-1])
        unmet_demand.append(CostFunction.normalized_unserved_energy(expected_energy, served_energy, power_outage=b.power_outage_signal)[-1])

    reward = (np.mean(np.array(resilience)) + np.mean(np.array(unmet_demand))) / 2

    if return_all:
        return -reward, np.array(resilience), np.array(unmet_demand)
    else:
        return -reward


def custom_outage_reward_per_step(env, comfort_band, return_all=False):
    """
    :param env: CityLearn environment
    :param comfort_band: comfort band for comfort reward
    :param step: env step to calculate reward
    :param return_all: return all values for debugging
    """
    resilience = []
    unmet_demand = []
    for b in env.buildings:
        assert b.power_outage_signal[-1] == 1

        expected_energy = b.cooling_demand[-1] + b.heating_demand[-1] + b.dhw_demand[-1] + b.non_shiftable_load[-1]
        served_energy = b.energy_from_cooling_device[-1] + b.energy_from_cooling_storage[-1] \
                        + b.energy_from_heating_device[-1] + b.energy_from_heating_storage[-1] \
                        + b.energy_from_dhw_device[-1] + b.energy_from_dhw_storage[-1] \
                        + b.energy_to_non_shiftable_load[-1]

        unserved_energy = (expected_energy - served_energy) / expected_energy

        indoor_dry_bulb_temperature = b.indoor_dry_bulb_temperature[-1]
        dry_bulb_temperature_set_point = b.indoor_dry_bulb_temperature_set_point[-1]
        occupant_count = b.occupant_count[-1]
        if occupant_count < 1.0:
            discomfort = 0
        else:
            delta = indoor_dry_bulb_temperature - dry_bulb_temperature_set_point
            if np.abs(delta) > comfort_band:
                discomfort = 1
            else:
                discomfort = 0

        resilience.append(discomfort)
        unmet_demand.append(unserved_energy)

    resilience = np.array(resilience)
    unmet_demand = np.array(unmet_demand)
    rewards = (resilience + unmet_demand) / 2
    if return_all:
        return -rewards, resilience, unmet_demand
    else:
        return -rewards


