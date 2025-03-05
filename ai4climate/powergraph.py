"""Load requested subtask for PowerGraph."""
import os
import gc
import json

import dataset_utils


# list of grid names
ALL_GRIDS_LIST = ['ieee24', 'ieee39', 'ieee118', 'uk']

# train validation testing ratio
SPLIT_RATIO = (0.5, 0.1, 0.4)


def load(
    local_dir: str, 
    subtask_name: str,
    data_frac: (int | float),
    train_frac: (int | float),
    max_workers: int
):
    """ """

    # load json files
    data_dict = _load_json_files(local_dir, data_frac)

    # parse into subtask datasets
    data_dict = _parse_dataset(data_dict)

    # shuffle data 
    data_dict = dataset_utils.shuffle_datadict(data_dict)

    # split into training, validation and testing
    train_data, val_data, test_data = _split_dataset(data_dict, train_frac)

    # clean up
    del data_dict
    gc.collect()

    # create dictionary for returning data
    subtask_data = {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data 
    }

    return subtask_data


def _parse_dataset(data_dict: dict):
    """ """

    return data_dict


def _split_dataset(
    data_dict: dict, 
    train_frac: float
) -> (dict, dict, dict):
    """ """

    # compute the splits
    total_size = len(data_dict)
    size_train = int(total_size * SPLIT_RATIO[0])
    size_val = int(total_size * SPLIT_RATIO[1])
    id_end_train = int(size_train * train_frac)
    id_start_val = size_train
    id_end_val = size_train + size_val

    # convert to list
    items = list(data_dict.items())
    
    # clean up
    del data_dict
    gc.collect()

    # split
    train_data = items[:id_end_train]
    val_data = items[id_start_val:id_end_val]
    test_data = items[id_end_val:]

    # clean up
    del items
    gc.collect()

    # return as dicts
    return dict(train_data), dict(val_data), dict(test_data)


def _load_json_files(
    local_dir: str, 
    data_frac: int
) -> dict:
    """ """

    # iterate over all datasets
    for gridname in ALL_GRIDS_LIST:
        print(f"Loading data for {gridname}")

        # set path to current grid
        path_grid = os.path.join(local_dir, gridname)

        # list all JSON files for grid
        file_list_grid = os.listdir(path_grid)

        # shorten file lis according to data_frac
        n_files_frac = data_frac * len(file_list_grid)
        file_list_grid = file_list_grid[:n_files_frac]

        # declare empty dictionary to load all data to it.
        data_dict = {}
        # iterate over all files
        for filename in file_list_grid:

            # load data only if json file
            if filename.endswith('.json'):

                # set path to current file
                path_file = os.path.join(path_grid, filename)

                # load data
                with open(path_file, 'r') as json_file:
                    f = json.load(json_file)

            else:
                continue

            # add to data dictionary
            data_dict.update(f)

    return data_dict 