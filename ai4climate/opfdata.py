import os
import gc
import math
import json
import random

import torch
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Tuple

import dataset_utils


SMALL_GRIDS_LIST = [
    'pglib_opf_case14_ieee',
    'pglib_opf_case30_ieee',
    'pglib_opf_case57_ieee'
]

MEDIUM_GRIDS_LIST = [
    'pglib_opf_case118_ieee',
    'pglib_opf_case500_goc',
    'pglib_opf_case2000_goc'
]

LARGE_GRIDS_LIST = [
    'pglib_opf_case4661_sdet',
    'pglib_opf_case6470_rte',
    'pglib_opf_case10000_goc',
    'pglib_opf_case13659_pegase'
]

SPLIT_RATIO = (0.5, 0.1, 0.4)  # train, val, test

# Set default dtypes
torch.set_default_dtype(torch.float64)
np_dtype = np.float64

def load(
    local_dir: str, 
    subtask_name: str,
    data_frac: float,
    train_frac: float,
    max_workers: int
):
    """
    Load out-of-distribution benchmark by merging grids in one pass.
    """

    # 1) Which grids to load
    if subtask_name.startswith('train_small'):
        train_grids = SMALL_GRIDS_LIST
    elif subtask_name.startswith('train_medium'):
        train_grids = MEDIUM_GRIDS_LIST
    elif subtask_name.startswith('train_large'):
        train_grids = LARGE_GRIDS_LIST
    else:
        raise ValueError(f"Unknown subtask_name: {subtask_name}")

    if subtask_name.endswith('test_small'):
        test_grids = SMALL_GRIDS_LIST
    elif subtask_name.endswith('test_medium'):
        test_grids = MEDIUM_GRIDS_LIST
    elif subtask_name.endswith('test_large'):
        test_grids = LARGE_GRIDS_LIST
    else:
        raise ValueError(f"Unknown subtask_name: {subtask_name}")

    # 2) Load train/val
    train_val_dataset = _load_multiple_grids(local_dir, train_grids, data_frac, max_workers)
    train_val_dataset = dataset_utils.shuffle_datadict(train_val_dataset)
    total_size = len(train_val_dataset)
    split_normalize = SPLIT_RATIO[0] + SPLIT_RATIO[1]
    train_ratio = SPLIT_RATIO[0] / split_normalize
    val_ratio = SPLIT_RATIO[1] / split_normalize
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

    # 3) Load test
    test_dataset = _load_multiple_grids(local_dir, test_grids, data_frac, max_workers)
    test_dataset = dataset_utils.shuffle_datadict(test_dataset)
    total_size = len(test_dataset)
    size_test = int(total_size * SPLIT_RATIO[2])
    test_items = list(test_dataset.items())[:size_test]
    test_dataset = dict(test_items)

    # 4) Parse data
    train_dataset, valid_dataset, test_dataset = _parse_datasets(
        train_dataset, 
        valid_dataset, 
        test_dataset,
    )

    # 5) Problem functions
    loss_functions = {
        'obj_gen_cost': obj_gen_cost,
        'eq_pbalance_re': eq_pbalance_re,
        'eq_pbalance_im': eq_pbalance_im, 
        'ineq_lower_box': ineq_lower_box,
        'ineq_upper_box': ineq_upper_box,
        'eq_difference': eq_difference
    }

    # 6) Return dictionary
    subtask_data = {
        'train_data': train_dataset,
        'val_data': valid_dataset,
        'test_data': test_dataset,
        'loss_functions': loss_functions,
    }
    return subtask_data


def _parse_datasets(
    train_dataset: dict,
    val_dataset: dict,
    test_dataset: dict
):
    """Turn raw dict-of-dict data into list-of-data-points with the requested backend."""
    train_dataset = _convert_dict_to_list(train_dataset)
    val_dataset   = _convert_dict_to_list(val_dataset)
    test_dataset  = _convert_dict_to_list(test_dataset)
    return train_dataset, val_dataset, test_dataset


def _convert_dict_to_list(dataset_dict: dict):
    data_list = []
    for i in range(len(dataset_dict)):
        _, v = dataset_dict.popitem()
        parsed = _parse_and_aggregate_datapoint(v, i_data=i)
        data_list.append(parsed)
    return data_list


def _parse_and_aggregate_datapoint(
    datapoint_dict: dict, 
    i_data: int
) -> dict:
    """Parse data point dictionary into features. Return either Torch or Numpy structures."""

    # Some dimension metadata
    baseMVA = datapoint_dict['grid']['context'][0][0][0]
    n = len(datapoint_dict['grid']['nodes']['bus'])
    n_e = (
        len(datapoint_dict['grid']['edges']['ac_line']['features'])
        + len(datapoint_dict['grid']['edges']['transformer']['features'])
    )
    n_g = len(datapoint_dict['grid']['nodes']['generator'])

    load_buses = datapoint_dict['grid']['edges']['load_link']['receivers']
    shunt_buses = datapoint_dict['grid']['edges']['shunt_link']['receivers']
    generator_buses = datapoint_dict['grid']['edges']['generator_link']['receivers']

    # --- Node-level ---
    (
        Sd_re_n, Sd_im_n, Ys_re_n, Ys_im_n, basekV_n, vl_n, vu_n, va_init_n, 
        ref_node_n, bustype_n, ref_bus, ref_count, vang_low_n, vang_up_n
    ) = _set_nodelevel_values(n, datapoint_dict, load_buses, shunt_buses)

    # --- Gen-level ---
    (
        Sl_re_g, Sl_im_g, Su_re_g, Su_im_g, c0_g, c1_g, c2_g, mbase_g,
        pg_init_g, qg_init_g, gen_bus_g, ref_gen_list, vm_init_n, Sgl_re_n,
        Sgl_im_n, Sgu_re_n, Sgu_im_n, c0g_n, c1g_n, c2g_n, num_gen_n
    ) = _set_generatorlevel_values(n, n_g, datapoint_dict, generator_buses, ref_bus)

    # --- Edge-level ---
    (
        ij_e, ijR_e, Y_re_e, Y_im_e, Yc_ij_im_e, Yc_ijR_im_e, T_mag_e,
        T_ang_e, su_e, vangl_e, vangu_e, Yc_ij_re_e, Yc_ijR_re_e, 
        Y_re_n, Y_im_n, sang_low_e, sang_up_e, suR_e, Y_mag_e, Y_ang_e
    ) = _set_edgelevel_values(n_e, datapoint_dict, Ys_re_n, Ys_im_n)

    # Collect some grid-level attributes
    grid_attr = {
        "baseMVA": baseMVA,
        "n": n,
        "n_e": n_e,
        "n_g": n_g,
        "ref_bus": ref_bus,
        "ref_gen_list": ref_gen_list,
        "bustype_n": bustype_n,
        "ij_e": ij_e,
        "ijR_e": ijR_e,
        "gen_bus_g": gen_bus_g,
    }

    # Convert these attributes if needed
    grid_attr = _to_numpy_dict(grid_attr)

    # Node feature matrix
    x_node = np.stack(
        [
            Sd_re_n, Sd_im_n, Ys_re_n, Ys_im_n, Y_re_n, Y_im_n,
            Sgl_re_n, Sgl_im_n, Sgu_re_n, Sgu_im_n, vl_n, vu_n,
            c0g_n, c1g_n, c2g_n, num_gen_n, ref_node_n, basekV_n
        ],
        axis=1
    )

    # Edge feature matrix
    x_edge = np.stack(
        [
            Y_re_e, Y_im_e, Yc_ij_re_e, Yc_ij_im_e, Yc_ijR_re_e,
            Yc_ijR_im_e, su_e, vangl_e, vangu_e, T_mag_e, T_ang_e
        ],
        axis = 1
    )

    # Generator feature matrix
    x_gen = np.stack(
        [
            Sl_re_g, Sl_im_g, Su_re_g, Su_im_g, c0_g, c1_g, c2_g, mbase_g
        ],
        axis=1
    )

    # Edge index (undirected: cat forward + backward)
    ei_np = np.concatenate((ij_e, ijR_e), axis=0)
    edge_index = ei_np.astype(np.int64)

    return {
        "x_node": x_node,
        "x_edge": x_edge,
        "x_gen": x_gen,
        "edge_index": edge_index,
        "grid_attr": grid_attr,
    }


def _load_multiple_grids(
    local_dir: str, 
    grid_list: list[str], 
    data_frac: float,
    max_workers: int
) -> dict:
    """Collect and parallel-load JSON data from all grids in grid_list."""
    all_json_paths = []
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
            n_sample_files = math.ceil(len(json_list) * data_frac)
            json_list = json_list[:n_sample_files]
            for fname in json_list:
                all_json_paths.append(os.path.join(path_group, fname))

    random.shuffle(all_json_paths)

    combined_dataset = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_read_json, fpath) for fpath in all_json_paths]
        for f in as_completed(futures):
            data_part = f.result()  # Dict from a single file
            combined_dataset.update(data_part)

    return combined_dataset


def _read_json(fpath: str) -> dict:
    with open(fpath, 'r') as fp:
        return json.load(fp)


def _set_nodelevel_values(
    n: int, 
    datapoint_dict: dict, 
    load_buses: dict, 
    shunt_buses: int
) -> Tuple:
    """ """
    Sd_re_n = np.zeros(n, dtype=np_dtype)
    Sd_im_n = np.zeros(n, dtype=np_dtype)
    Ys_re_n = np.zeros(n, dtype=np_dtype)
    Ys_im_n = np.zeros(n, dtype=np_dtype)
    vl_n    = np.zeros(n, dtype=np_dtype)
    vu_n    = np.zeros(n, dtype=np_dtype)
    ref_node_n = np.ones(n, dtype=np.intc)
    bustype_n  = np.zeros(n, dtype=np.intc)
    basekV_n   = np.zeros(n, dtype=np.intc)
    va_init_n  = np.zeros(n, dtype=np_dtype)

    ref_bus = None
    ref_count = 0

    for idx, values in enumerate(datapoint_dict['grid']['nodes']['bus']):
        basekV_n[idx]  = values[0]
        bustype_n[idx] = values[1]  # PQ(1), PV(2), REF(3), etc.
        vl_n[idx]      = values[2]  # p.u.
        vu_n[idx]      = values[3]
        if bustype_n[idx] == 3:  # reference bus
            ref_node_n[idx] = 0
            ref_bus = idx
            ref_count += 1
    
    for idx, values in enumerate(datapoint_dict['grid']['nodes']['load']):
        Sd_re_n[load_buses[idx]] += values[0]
        Sd_im_n[load_buses[idx]] += values[1]
        
    for idx, values in enumerate(datapoint_dict['grid']['nodes']['shunt']):
        Ys_im_n[shunt_buses[idx]] += values[0]
        Ys_re_n[shunt_buses[idx]] += values[1]
        
    vang_low_n = -np.pi / 2 * np.ones((n), dtype=np_dtype)
    vang_up_n  =  np.pi / 2 * np.ones((n), dtype=np_dtype)

    return (
        Sd_re_n, Sd_im_n, Ys_re_n, Ys_im_n, basekV_n, vl_n, vu_n, va_init_n, 
        ref_node_n, bustype_n, ref_bus, ref_count, vang_low_n, vang_up_n
    )


def _set_generatorlevel_values(
    n, 
    n_g, 
    datapoint_dict, 
    generator_buses, 
    ref_bus
) -> Tuple:
    vm_init_n  = np.ones(n, dtype=np_dtype)
    Sgl_re_n   = np.zeros(n, dtype=np_dtype)
    Sgl_im_n   = np.zeros(n, dtype=np_dtype)
    Sgu_re_n   = np.zeros(n, dtype=np_dtype)
    Sgu_im_n   = np.zeros(n, dtype=np_dtype)
    c0g_n      = np.zeros(n, dtype=np_dtype)
    c1g_n      = np.zeros(n, dtype=np_dtype)
    c2g_n      = np.zeros(n, dtype=np_dtype)
    num_gen_n  = np.zeros(n, dtype=np.intc)

    Sl_re_g = np.zeros(n_g, dtype=np_dtype)
    Sl_im_g = np.zeros(n_g, dtype=np_dtype)
    Su_re_g = np.zeros(n_g, dtype=np_dtype)
    Su_im_g = np.zeros(n_g, dtype=np_dtype)
    c0_g    = np.zeros(n_g, dtype=np_dtype)
    c1_g    = np.zeros(n_g, dtype=np_dtype)
    c2_g    = np.zeros(n_g, dtype=np_dtype)
    mbase_g = np.zeros(n_g, dtype=np_dtype)
    pg_init_g = np.zeros(n_g, dtype=np_dtype)
    qg_init_g = np.zeros(n_g, dtype=np_dtype)

    gen_bus_g = np.array(generator_buses, dtype=np.intc)
    ref_gen_list = []

    for idx, values in enumerate(datapoint_dict['grid']['nodes']['generator']):
        mbase_g[idx]   = values[0]
        pg_init_g[idx] = values[1]
        Sl_re_g[idx]   = values[2]
        Su_re_g[idx]   = values[3]
        qg_init_g[idx] = values[4]
        Sl_im_g[idx]   = values[5]
        Su_im_g[idx]   = values[6]
        vm_init_n[gen_bus_g[idx]] = values[7]
        c2_g[idx]      = values[8]
        c1_g[idx]      = values[9]
        c0_g[idx]      = values[10]

        if gen_bus_g[idx] == ref_bus:
            ref_gen_list.append(idx)

    ref_gen_list = np.array(ref_gen_list, dtype=np.intc)

    for gen_id, node_id in enumerate(gen_bus_g):
        Sgl_re_n[node_id] += Sl_re_g[gen_id]
        Sgl_im_n[node_id] += Sl_im_g[gen_id]
        Sgu_re_n[node_id] += Su_re_g[gen_id]
        Sgu_im_n[node_id] += Su_im_g[gen_id]
        c0g_n[node_id]    += c2_g[gen_id]
        c1g_n[node_id]    += c1_g[gen_id]
        c2g_n[node_id]    += c0_g[gen_id]
        num_gen_n[node_id] += 1

    return (
        Sl_re_g, Sl_im_g, Su_re_g, Su_im_g, c0_g, c1_g, c2_g, mbase_g,
        pg_init_g, qg_init_g, gen_bus_g, ref_gen_list, vm_init_n,
        Sgl_re_n, Sgl_im_n, Sgu_re_n, Sgu_im_n, c0g_n, c1g_n, c2g_n,
        num_gen_n
    )


def _set_edgelevel_values(
    n_e, 
    datapoint_dict, 
    Ys_re_n, 
    Ys_im_n
) -> Tuple:
    # line buses
    ij_line = np.column_stack(
        (
            datapoint_dict['grid']['edges']['ac_line']['senders'], 
            datapoint_dict['grid']['edges']['ac_line']['receivers']
        )
    )
    # transformer buses
    ij_transformer = np.column_stack(
        (
            datapoint_dict['grid']['edges']['transformer']['senders'], 
            datapoint_dict['grid']['edges']['transformer']['receivers']
        )
    )

    ij_e = np.vstack((ij_line, ij_transformer)).astype(np.intc)
    Y_re_e = np.zeros(n_e, dtype=np_dtype)
    Y_im_e = np.zeros(n_e, dtype=np_dtype)
    Yc_ij_im_e = np.zeros(n_e, dtype=np_dtype)
    Yc_ijR_im_e = np.zeros(n_e, dtype=np_dtype)
    T_mag_e = np.ones(n_e, dtype=np_dtype)
    T_ang_e = np.zeros(n_e, dtype=np_dtype)
    su_e = np.zeros(n_e, dtype=np_dtype)
    vangl_e = np.zeros(n_e, dtype=np_dtype)
    vangu_e = np.zeros(n_e, dtype=np_dtype)

    Yc_ij_re_e  = np.zeros(n_e, dtype=np_dtype)
    Yc_ijR_re_e = np.zeros(n_e, dtype=np_dtype)

    # lines
    idx = -1
    for line_vals in datapoint_dict['grid']['edges']['ac_line']['features']:
        idx += 1
        vangl_e[idx]     = line_vals[0]
        vangu_e[idx]     = line_vals[1]
        Yc_ij_im_e[idx]  = line_vals[2]
        Yc_ijR_im_e[idx] = line_vals[3]
        r = line_vals[4]
        x = line_vals[5]
        Y_re_e[idx] = r / (r**2 + x**2)
        Y_im_e[idx] = -x / (r**2 + x**2)
        su_e[idx]   = line_vals[6]

    # transformers
    for transf_vals in datapoint_dict['grid']['edges']['transformer']['features']:
        idx += 1
        vangl_e[idx]     = transf_vals[0]
        vangu_e[idx]     = transf_vals[1]
        r = transf_vals[2]
        x = transf_vals[3]
        Y_re_e[idx] = r / (r**2 + x**2)
        Y_im_e[idx] = -x / (r**2 + x**2)
        su_e[idx]   = transf_vals[4]
        T_mag_e[idx] = transf_vals[7]
        T_ang_e[idx] = transf_vals[8]
        Yc_ij_im_e[idx]  = transf_vals[9]
        Yc_ijR_im_e[idx] = transf_vals[10]

    # Convert Y from rectangular to polar
    Y_mag_e, Y_ang_e = _rectangle_to_polar(Y_re_e, Y_im_e)

    # Reverse edge indices
    ijR_e = ij_e[:, [1, 0]]

    # Build up node admittance
    Y_re_n_new = Ys_re_n.copy()
    Y_im_n_new = Ys_im_n.copy()
    for branch_k in range(len(ij_e)):
        i_node = ij_e[branch_k, 0]
        j_node = ij_e[branch_k, 1]
        Y_re_n_new[i_node] += Y_re_e[branch_k]
        Y_re_n_new[j_node] += Y_re_e[branch_k]
        Y_im_n_new[i_node] += Y_im_e[branch_k]
        Y_im_n_new[j_node] += Y_im_e[branch_k]

    sang_low_e = -np.pi / 2 * np.ones(n_e, dtype=np_dtype)
    sang_up_e  =  np.pi / 2 * np.ones(n_e, dtype=np_dtype)

    # same forward / reverse limit
    suR_e = su_e.copy()

    return (
        ij_e, ijR_e, Y_re_e, Y_im_e, Yc_ij_im_e, Yc_ijR_im_e, T_mag_e,
        T_ang_e, su_e, vangl_e, vangu_e, Yc_ij_re_e, Yc_ijR_re_e, 
        Y_re_n_new, Y_im_n_new, sang_low_e, sang_up_e, suR_e,
        Y_mag_e, Y_ang_e
    )


def _rectangle_to_polar(X_re, X_im):
    """
    Transform from rectangular to polar form, skipping zero-div by a small offset.
    This version stays in NumPy by default; we only convert to Torch if needed.
    """
    small_number = 1.e-10

    # We can do the calculation in NumPy first:
    X_mag_np = np.sqrt(X_re**2 + X_im**2)
    X_ang_np = np.arctan(X_im / (X_re + small_number))

    return X_mag_np, X_ang_np


def _to_numpy_dict(data_dict: dict):
    """
    Convert each array-like or scalar in data_dict.
    """
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            # We assume it's already np, maybe ensure float64 or int
            pass
        elif isinstance(value, (int, float)):
            data_dict[key] = np.array([value], dtype=np_dtype)

    return data_dict


def obj_gen_cost(Sg_re_g, c2_g, c1_g, c0_g):
    return c2_g * Sg_re_g**2 + c1_g * Sg_re_g + c0_g

def eq_pbalance_re(Sg_re_n, Sd_re_n, Ys_re_n, V_mag_n, Sij_re_n, SijR_re_n):
    return Sg_re_n - Sd_re_n - Ys_re_n * V_mag_n**2 - Sij_re_n - SijR_re_n

def eq_pbalance_im(Sg_im_n, Sd_im_n, Ys_im_n, V_mag_n, Sij_im_n, SijR_im_n):
    return Sg_im_n - Sd_im_n + Ys_im_n * V_mag_n**2 - Sij_im_n - SijR_im_n

def ineq_lower_box(x_value, x_lower):
    return x_lower - x_value

def ineq_upper_box(x_value, x_upper):
    return x_value - x_upper

def eq_difference(x_value, x_true_value):
    return x_value - x_true_value