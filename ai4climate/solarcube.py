"""Loads requested subtask for SolarCube.

"""
from typing import Dict, Any, Tuple, Union, List
import os
import h5py
import pandas as pd
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed

# number of metorological stations and geographic areas.
NUMBER_STATIONS = 19

SAT_IMAGE_NAME_LIST = [
    'cloud_mask',
    'infrared_band_133',
    'satellite_solar_radiation',
    'solar_zenith_angle',
    'visual_band_47',
    'visual_band_86'
]

LIST_AVAIL_SUBTASKNAMES = [
    'odd_time_area_3h',
    'odd_time_area_24h',
    'odd_time_point_3h',
    'odd_time_point_24h',
    'odd_space_area_3h',
    'odd_space_area_24h',
    'odd_space_point_3h',
    'odd_space_point_24h',
    'odd_spacetime_area_3h',
    'odd_spacetime_area_24h',
    'odd_spacetime_point_3h',
    'odd_spacetime_point_24h'
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
    if subtask_name not in LIST_AVAIL_SUBTASKNAMES:
        raise ValueError(f"Unknown subtask name: {subtask_name}")
        
    # load csv data
    station_features, ground_radiation = _load_csv_data(local_dir, subtask_name)

    # load hiearchical data format files
    satellite_images_dict = _load_hdf5_data(local_dir, subtask_name, max_workers)

    train_data = 0
    val_data = 0
    test_data = 0

    return {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data
    }


def _load_csv_data(
    local_dir: str, 
    subtask_name: str
):
    """
    Load csv data files as dataframes.

    """
    # load station features
    path_station_features = os.path.join(local_dir, 'station_features.csv')  
    station_features = pd.read_csv(path_station_features)
    
    # load ground radiation only for point based prediction subtasks.
    if 'point' in subtask_name:
        path_ground_radiation = os.path.join(local_dir, 'ground_radiation.csv')  
        ground_radiation = pd.read_csv(path_ground_radiation)
    else:
        ground_radiation = None


    return station_features, ground_radiation


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
