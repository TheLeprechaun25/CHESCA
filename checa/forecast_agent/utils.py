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


def fast_closest_hour(hour, hours_with_data):
    closest_hour_with_data = 0
    closeness = hour
    for h in hours_with_data:
        if abs(hour - h) < closeness:
            closeness = abs(hour - h)
            closest_hour_with_data = h
    return closest_hour_with_data


def get_closest_hour_data(df, target_hour):
    """Helper function to find the row with the closest hour to the target_hour when there's no exact match."""
    if not df[df['hour'] == target_hour].empty:
        return df[df['hour'] == target_hour]
    else:
        # Find the closest hour by calculating the minimal time difference
        closest_hour_row = df.iloc[(df['hour'] - target_hour).abs().argsort()[:1]]
        return closest_hour_row




