import numpy as np


def get_observation_names_with_building(observation_names):
    """
    Returns the observation names for a specific building
    """
    observation_names_by_building = observation_names.copy()
    for i, name in enumerate(observation_names_by_building):
        ocu = 0
        for j in range(i+1, len(observation_names_by_building)):
            if name == observation_names_by_building[j]:
                ocu += 1
                observation_names_by_building[j] = f"{name}_{ocu}"
                observation_names_by_building[i] = f"{name}_{0}"

    return observation_names_by_building


