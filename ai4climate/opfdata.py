"""Load requested subtask for OPFData."""
import os 
import gc
import math
import json
import random

from concurrent.futures import ThreadPoolExecutor, as_completed

small_grids_list =[
    'pglib_opf_case14_ieee',
    'pglib_opf_case30_ieee',
    'pglib_opf_case57_ieee'
]

medium_grids_list = [
    'pglib_opf_case118_ieee',
    'pglib_opf_case500_goc',
#    'pglib_opf_case2000_goc'
]

large_grids_list = [
    'pglib_opf_case4661_sdet',
    'pglib_opf_case6470_rte',
    'pglib_opf_case10000_goc',
    'pglib_opf_case13659_pegase'
]

# train validation testing ratio
split_ratio = (0.5, 0.1, 0.4)

def load(
    local_dir: str, 
    subtask_name: str,
    data_frac: int,
    train_frac: int,
    max_workers: int
):
    """Load out-of-distribution benchmark by merging grids in one pass."""
    
    # set train and validation grids
    if subtask_name.startswith('train_small'):
        train_grids = small_grids_list
    elif subtask_name.startswith('train_medium'):
        train_grids = medium_grids_list
    elif subtask_name.startswith('train_large'):
        train_grids = large_grids_list
    
    # set testing grids
    if subtask_name.endswith('test_small'):
        test_grids = small_grids_list
    elif subtask_name.endswith('test_medium'):
        test_grids = medium_grids_list
    elif subtask_name.endswith('test_large'):
        test_grids = large_grids_list

    # 1) Training and Validation datasets
    train_val_dataset = _load_multiple_grids(local_dir, train_grids, 
        data_frac, max_workers)
    train_val_dataset = _shuffle_datadict(train_val_dataset)

    total_size = len(train_val_dataset)
    split_normalize = split_ratio[0] + split_ratio[1]
    train_ratio = split_ratio[0] / split_normalize
    val_ratio = split_ratio[1] / split_normalize
    size_train = int(total_size * train_ratio * train_frac)
    size_val = int(total_size * val_ratio)

    items = list(train_val_dataset.items())
    del train_val_dataset
    gc.collect()

    train_items = items[:size_train]
    val_items = items[size_train : size_train + size_val]
    del items
    gc.collect()

    train_dataset = dict(train_items)
    valid_dataset = dict(val_items)

    # 2) Gather and load all test grids
    test_dataset = _load_multiple_grids(local_dir, test_grids, data_frac, 
        max_workers)
    test_dataset = _shuffle_datadict(test_dataset)

    total_size = len(test_dataset)
    size_test = int(total_size * split_ratio[2])

    test_items = list(test_dataset.items())
    test_items = test_items[:size_test]
    test_dataset = dict(test_items)
    
    # 3) Parse data


    return train_dataset, valid_dataset, test_dataset


def _load_multiple_grids(
    local_dir: str,
    grid_list: list[str],
    data_frac: float,
    max_workers: int
) -> dict:
    """Collect and parallel-load JSON data from all grids in grid_list."""
    all_json_paths = []
    
    # Collect file paths from each grid
    for gridname in grid_list:
        path_grid = os.path.join(local_dir, gridname)
        group_list = [
            g for g in os.listdir(path_grid) if g.startswith('group')
        ]
        random.shuffle(group_list)
        
        for group in group_list:
            path_group = os.path.join(path_grid, group)
            json_list = [
                fname for fname in os.listdir(path_group)
                if fname.endswith('.json')
            ]
            random.shuffle(json_list)
            
            # Subsample by data_frac
            n_sample_files = math.ceil(len(json_list) * data_frac)
            json_list = json_list[:n_sample_files]
            
            # Accumulate full paths
            for fname in json_list:
                all_json_paths.append(os.path.join(path_group, fname))

    # Shuffle all file paths once
    random.shuffle(all_json_paths)
    
    # Parallel load
    combined_dataset = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_read_json, fpath) for fpath in all_json_paths]
        for f in as_completed(futures):
            data_part = f.result()  # Dict from a single file
            combined_dataset.update(data_part)
    
    return combined_dataset

def _read_json(fpath: str) -> dict:
    """Helper to read JSON and return it as a Python dict."""
    with open(fpath, 'r') as fp:
        return json.load(fp)

def _shuffle_datadict(dataset):
    """Shuffle a dictionary by key."""
    items = list(dataset.items())
    random.shuffle(items)
    return dict(items)