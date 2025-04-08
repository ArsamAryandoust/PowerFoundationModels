"""Loads requested subtask for BuildingElectricity.

"""
import gc
import os
import h5py
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Union, List
from concurrent.futures import ThreadPoolExecutor, as_completed

AVAIL_SUBTASKNAMES_LIST = [
    'odd_time_buildings92',
    'odd_space_buildings92',
    'odd_spacetime_buildings92',
    'odd_time_buildings451',
    'odd_space_buildings451',
    'odd_spacetime_buildings451'
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

    Parameters
    ----------
    local_dir : str
        Local path containing the subtask data.
    subtask_name : str
        One of the recognized subtask names (see AVAIL_SUBTASKNAMES_LIST).
    data_frac : Union[int, float]
        Overall fraction of samples to keep from full dataset.
    train_frac : Union[int, float]
        Fraction of the standardized training split to actually use.
    max_workers : int
        Number of parallel workers for loading data from HDF5.
    seed : int, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        A dictionary with keys ['train_data', 'val_data', 'test_data'].

    """
    if subtask_name not in AVAIL_SUBTASKNAMES_LIST:
        raise ValueError(f"Unknown subtask name: {subtask_name}")

    # exdend local_dir with corresponding profiles
    if subtask_name.endswith('92'):
        local_dir = os.join.path(local_dir, 'profiles_92')
    elif subtask_name.endswith('451'):
        local_dir = os.join.path(local_dir, 'profiles_451')
    else:
        raise VallueError('Check subtask handling. Naming not consistent!')

    # load electric profiles
    df_loads = load_electric_load_profiles(local_dir)

    

    subtask_data = {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data
    }
    

def load_electric_load_profiles(local_dir: str):
    """
    Load electric load profiles as DataFrame

    """

    # load load profiles
    path_load = os.path.join(local_dir, 'load_profiles.csv')
    df_loads = pd.read_csv(path_load)

    return df_loads 