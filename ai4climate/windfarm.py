"""Loads requested subtask for WindFarm.

"""
from typing import Dict, Any, Tuple, Union, List

AVAIL_SUBTASKNAMES_LIST = [
    'odd_time_predict48h',
    'odd_space_predict48h',
    'odd_spacetime_predict48h',
    'odd_time_predict72h',
    'odd_space_predict72h',
    'odd_spacetime_predict72h'
]


# The standardized split ratio for the entire dataset: train, val, test
SPLIT_RATIO = (0.5, 0.1, 0.4)

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

    