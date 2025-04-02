"""Loads requested subtask for SolarCube.

"""
from typing import Dict, Any, Tuple, Union, List
import gc
import os
import h5py
import pandas as pd
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed

import sys

# number of metorological stations and geographic areas.
NUMBER_STATIONS = 19

# past and future time window of 24h in 15 minute steps equals 96 time steps.
TIME_WINDOW_STEPS = 96

# list of satellite data available to load
SAT_IMAGE_NAME_LIST = [
    'cloud_mask',
    'infrared_band_133',
    'satellite_solar_radiation',
    'solar_zenith_angle',
    'visual_band_47',
    'visual_band_86'
]

# list of available subtask datasets available to load.
AVAIL_SUBTASKNAMES_LIST = [
    'odd_time_area',
    'odd_time_point',
    'odd_space_area',
    'odd_space_point',
    'odd_spacetime_area',
    'odd_spacetime_point'
]

def load(
    local_dir: str,
    subtask_name: str,
    data_frac: Union[int, float],
    train_frac: Union[int, float],
    max_workers: int,
    seed: int = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load and prepare the data for a given subtask.

    """
    # check if valid subtask name is passed
    if subtask_name not in AVAIL_SUBTASKNAMES_LIST:
        raise ValueError(f"Unknown subtask name: {subtask_name}")
        
    # load csv data
    timestamps_dict, station_features_df, ground_radiation_df = _load_csv_data(
        local_dir, 
        subtask_name
    )

    # load hiearchical data format files
    satellite_images_dict = _load_hdf5_data(local_dir, subtask_name, max_workers)

    # create feature label pairs forming dataset
    features, labels = _process_features_labels(
        timestamps_dict,
        station_features_df, 
        ground_radiation_df,
        satellite_images_dict,
        subtask_name
    )

    # free memory
    del timestamps_dict, station_features_df, ground_radiation_df
    del satellite_images_dict
    gc.collect()

    # create paired dataset
    paired_dataset = _pair_features_labels(features, labels)

    # free memory
    del features, labels
    gc.collect()

    # create train-val-test splits
    train_data, val_data, test_data = _split_data(
        features, 
        labels,
        data_frac, 
        train_frac,
        subtask_name
    )

    # free memory
    del paired_dataset
    gc.collect()

    return {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data
    }


def _process_features_labels(
    timestamps_dict: dict,
    station_features_df: pd.DataFrame,
    ground_radiation_df: pd.DataFrame | None,
    satellite_images_dict: dict,
    subtask_name: str
):
    """
    Create features and labels for supervised learning dataset of subtask.

    """
    ### fill with features and labels
    features = {}
    labels = {}

    ### iterate over all stations and pair timestamp, satellite and ground data. 
    for id, (key, value) in enumerate(satellite_images_dict.items()):

        ### add time stamp data
        timestamp_utc = timestamps_dict[id+1]['utc_time'].values.tolist()
        timestamp_local = timestamps_dict[id+1]['local_time'].values.tolist()

        ### add location data
        latitude = station_features_df['lats'][id]
        longitude = station_features_df['lons'][id]
        elevation = station_features_df['elev'][id]

        if ground_radiation_df is not None:
            ### this is the case for point level predictions.

            # define space and time variant features
            features_satellite = np.stack(
                [
                    value['infrared_band_133'],
                    value['visual_band_47'],
                    value['visual_band_86'],
                    value['solar_zenith_angle']
                ], 
                axis=-1
            )

            # set ground measurement features
            features_ground = ground_radiation_df[str(id+1)].to_numpy()

            # set labels
            label_task = ground_radiation_df[str(id+1)].to_numpy()

        else:
            ### this is the case for area level predictions.

            # define space and time variant features
            features_satellite = np.stack(
                [
                    value['infrared_band_133'],
                    value['visual_band_47'],
                    value['visual_band_86'],
                    value['solar_zenith_angle'],
                    value['cloud_mask'],
                    value['satellite_solar_radiation']
                ], 
                axis=-1
            )

            # set to None for consistency of feature structure.
            features_ground = None

            # set labels. satellite solar radiation multiplied with cloud mask.
            label_task = (
                satellite_images_dict['satellite_solar_radiation'] 
                * satellite_images_dict['cloud_mask']
            )

        ### add to features dictionary
        features[id+1] = {
            'timestamp_utc': timestamp_utc,
            'timestamp_local': timestamp_local,
            'latitude': latitude,
            'longitude': longitude,
            'elevation': elevation,
            'features_satellite': features_satellite,
            'features_ground': features_ground
        }

        ### add to labels dictionary
        labels[id+1] = label_task

    return features, labels


def _pair_features_labels(
    features: dict,
    labels: dict
) -> dict:
    """
    Pair features and labels into single dataset.

    """
    
    paired_dataset = {}

    return paired_dataset
 
def _split_data(
    features: dict,
    labels: dict,
    data_frac: Union[int, float],
    train_frac: Union[int, float],
    subtask_name: str
):
    """ 
    Split feature 
    """


    return train_data, val_data, test_data


def _load_csv_data(
    local_dir: str, 
    subtask_name: str
) -> (dict, pd.DataFrame, pd.DataFrame | None):
    """
    Load csv data files as dataframes.

    """
    ### load time stamp data
    path_timestamps = os.path.join(local_dir, 'availability_IDs')
    timestamps_dict = {}

    for i in range(NUMBER_STATIONS):
        filename = f'station_{i+1}_timestamps.csv'
        path_timefile = os.path.join(path_timestamps, filename)
        timestamps = pd.read_csv(path_timefile)
        timestamps_dict[i+1] = timestamps

    ### load station features
    path_station_features_df = os.path.join(local_dir, 'station_features.csv')  
    station_features_df = pd.read_csv(path_station_features_df)
    
    ### load ground radiation only for point-based prediction subtasks.
    if 'point' in subtask_name:
        path_ground_radiation_df = os.path.join(local_dir, 'ground_radiation.csv')  
        ground_radiation_df = pd.read_csv(path_ground_radiation_df)
    else:
        ground_radiation_df = None

    return timestamps_dict, station_features_df, ground_radiation_df


def _load_hdf5_data(
    local_dir: str, 
    subtask_name: str,
    max_workers: int
):
    """
    Load HDF5 data files as numpy arrays.

    """

    # define helper function for parallel loading
    def load_helper(local_dir, i):
        # set station directory name
        directory_name = f'station_{i+1}'

        # set path to station directory
        path_station = os.path.join(local_dir, directory_name)

        # fill dictionary with this station's data
        station_images_dict = {}

        # iterate over all names
        for image_name in SAT_IMAGE_NAME_LIST:
            # set path to file
            path_file = os.path.join(path_station, image_name + '.h5')

            # load file as numpy array
            data = h5py.File(path_file, 'r').get(image_name)[:]

            # append file to station data dict
            station_images_dict[image_name] = data

        return directory_name, station_images_dict

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                load_helper, local_dir, i
            ) for i in range(NUMBER_STATIONS)
        ]

    # fill dictionary with all results
    satellite_images_dict = {}

    # iterate over all results
    for future in as_completed(futures):

        directory_name, data = future.result()

        # append results to final dictionary
        satellite_images_dict[directory_name] = data

    return satellite_images_dict
