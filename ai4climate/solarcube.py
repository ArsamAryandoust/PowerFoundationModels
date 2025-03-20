"""Loads requested subtask for SolarCube.

"""
from typing import Dict, Any, Tuple, Union, List

SPLIT_RATIO = (0.5, 0.1, 0.4)  # train, val, test

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
    train_data = 0
    val_data = 0
    test_data = 0

    return {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data
    }

