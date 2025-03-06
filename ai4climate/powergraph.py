"""Load requested subtask for PowerGraph."""
import os
import gc
import json
import logging
from typing import Dict, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import dataset_utils


# list of grid names
ALL_GRIDS_LIST = ['ieee24', 'ieee39', 'ieee118', 'uk']

# train validation testing ratio
SPLIT_RATIO = (0.5, 0.1, 0.4)


def load(
    local_dir: str,
    subtask_name: str,
    data_frac: Union[int, float],
    train_frac: Union[int, float],
    max_workers: int = 1
) -> Dict[str, Dict[str, Any]]:
    """Load and prepare the data for a given subtask."""

    # load json files
    data_dict = _load_json_files(local_dir, data_frac, max_workers=max_workers)

    # parse into subtask datasets
    data_dict = _parse_dataset(data_dict)

    # shuffle data 
    data_dict = dataset_utils.shuffle_datadict(data_dict)

    # split into training, validation and testing
    train_data, val_data, test_data = _split_dataset(data_dict, train_frac)

    # Clean up large dictionary to free memory
    del data_dict
    gc.collect()

    return {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data
    }


def _parse_dataset(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """ """

    return data_dict


def _split_dataset(
    data_dict: Dict[str, Any],
    train_frac: float
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Split the dataset into train, validation, and test sets, based on 
    SPLIT_RATIO. Then within the train-slice, only a fraction (train_frac) is 
    used for actual training, discarding the remainder of the 'train slice'.

    """
    total_size = len(data_dict)
    size_train = int(total_size * SPLIT_RATIO[0])
    size_val = int(total_size * SPLIT_RATIO[1])

    id_end_train = int(size_train * train_frac)
    id_start_val = size_train
    id_end_val = size_train + size_val

    # Convert dict to list of (key, value) pairs for slicing
    items = list(data_dict.items())
    
    # Free the original dict from memory
    del data_dict
    gc.collect()

    # Slice the data
    train_data = dict(items[:id_end_train])
    val_data = dict(items[id_start_val:id_end_val])
    test_data = dict(items[id_end_val:])

    return train_data, val_data, test_data


def _load_json_files(
    local_dir: str,
    data_frac: Union[int, float],
    max_workers: int = 1
) -> Dict[str, Any]:
    """
    Load JSON files from each grid in ALL_GRIDS_LIST, accumulating them into a 
    single dictionary.

    """
    combined_data_dict = {}

    # For each grid, load its JSON files into the combined dictionary
    for gridname in ALL_GRIDS_LIST:
        path_grid = os.path.join(local_dir, gridname)
        if not os.path.isdir(path_grid):
            logging.warning(f"Directory '{path_grid}' does not exist, skipped.")
            continue

        # Gather all JSON files
        file_list_grid = [f for f in os.listdir(path_grid) if f.endswith('.json')]
        total_files = len(file_list_grid)

        if total_files == 0:
            logging.info(f"No JSON files found in '{path_grid}', skipped.")
            continue

        # compute how many files to load based on data_frac
        num_to_load = int(round(data_frac * total_files))

        file_list_grid = file_list_grid[:num_to_load]
        logging.info(f"Loading {num_to_load} files from '{path_grid}'.")

        def _read_json_file(filename: str) -> Dict[str, Any]:
            with open(filename, 'r', encoding='utf-8') as fh:
                return json.load(fh)

        # Load in parallel
        partial_dicts = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_read_json_file, os.path.join(path_grid, fname)): fname
                for fname in file_list_grid
            }
            for future in as_completed(future_map):
                fname = future_map[future]
                try:
                    file_data = future.result()
                    partial_dicts.append(file_data)
                except Exception as e:
                    logging.error(f"Error reading '{fname}': {e}")

        # Merge each file's dictionary into the global data structure
        for partial_dict in partial_dicts:
            combined_data_dict.update(partial_dict)

        # Clean up
        del partial_dicts
        gc.collect()

    return combined_data_dict