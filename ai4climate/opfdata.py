"""Load requested subtask for OPFData."""
import os 
import gc
import math
import json
import random
import torch
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed

small_grids_list =[
    'pglib_opf_case14_ieee',
    'pglib_opf_case30_ieee',
    'pglib_opf_case57_ieee'
]

medium_grids_list = [
    'pglib_opf_case118_ieee',
    'pglib_opf_case500_goc',
    'pglib_opf_case2000_goc'
]

large_grids_list = [
    'pglib_opf_case4661_sdet',
    'pglib_opf_case6470_rte',
    'pglib_opf_case10000_goc',
    'pglib_opf_case13659_pegase'
]

# train validation testing ratio
split_ratio = (0.5, 0.1, 0.4)

torch.set_default_dtype(torch.float64)
np_dtype = np.float64

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
    (
        train_dataset, 
        valid_dataset, 
        test_dataset
    ) = _parse_datasets(
        train_dataset, 
        valid_dataset, 
        test_dataset
    )

    # return as dictionary.
    subtask_data = {
        'train_data': train_dataset,
        'val_data': valid_dataset,
        'test_data': test_dataset
    }

    return subtask_data

def _parse_datasets(
    train_dataset: dict,
    val_dataset: dict,
    test_dataset: dict
) -> (dict, dict, dict):
    """ """

    # create empty list to fill datasets
    dataset_list = []
    # iterate over all datapoints in list of dictionaries
    for i in range(len(train_dataset)):
        # pop dictionary item
        _, v = train_dataset.popitem()
        # parse and aggregate data point
        v = _parse_and_aggregate_datapoint(v, i)
        # add parsed data point to dataset list
        dataset_list.append(v)
    # replace dataset in-place with parsed list
    train_dataset = dataset_list


    # create empty list to fill datasets
    dataset_list = []
    # iterate over all datapoints in list of dictionaries
    for i in range(len(val_dataset)):
        # pop dictionary item
        _, v = val_dataset.popitem()
        # parse and aggregate data point
        v = _parse_and_aggregate_datapoint(v, i)
        # add parsed data point to dataset list
        dataset_list.append(v)
    # replace dataset in-place with parsed list
    val_dataset = dataset_list

    # create empty list to fill datasets
    dataset_list = []
    # iterate over all datapoints in list of dictionaries
    for i in range(len(test_dataset)):
        # pop dictionary item
        _, v = test_dataset.popitem()
        # parse and aggregate data point
        v = _parse_and_aggregate_datapoint(v, i)
        # add parsed data point to dataset list
        dataset_list.append(v)
    # replace dataset in-place with parsed list
    test_dataset = dataset_list

    return train_dataset, val_dataset, test_dataset


def _parse_and_aggregate_datapoint(
    datapoint_dict:dict, 
    i_data:int
) -> dict:
    """Parse data point dictionary and aggregate into feature components."""

    # set graph-level values
    baseMVA = datapoint_dict['grid']['context'][0][0][0]
    n = len(datapoint_dict['grid']['nodes']['bus'])
    n_e = (
        len(datapoint_dict['grid']['edges']['ac_line']['features'])
        + len(datapoint_dict['grid']['edges']['transformer']['features'])
    )
    n_g = len(datapoint_dict['grid']['nodes']['generator'])
    
    # get generator, load, and shunt buses
    load_buses = datapoint_dict['grid']['edges']['load_link']['receivers']
    shunt_buses = datapoint_dict['grid']['edges']['shunt_link']['receivers']
    generator_buses = (
        datapoint_dict['grid']['edges']['generator_link']['receivers']
    )
    
    # set node-level values.
    (
        Sd_re_n, Sd_im_n, Ys_re_n, Ys_im_n, vl_n, vu_n, va_init_n, 
        ref_node_n, bustype_n, ref_bus, ref_count, vang_low_n, vang_up_n
    ) = _set_nodelevel_values(n, datapoint_dict, load_buses, 
        shunt_buses)

    # set generator-level values.
    (
        Sl_re_g, Sl_im_g, Su_re_g, Su_im_g, c0_g, c1_g, c2_g, mbase_g,
        pg_init_g, qg_init_g, gen_bus_g, ref_gen_list, vm_init_n, Sgl_re_n,
        Sgl_im_n, Sgu_re_n, Sgu_im_n, c0g_n, c1g_n, c2g_n, num_gen_n
    ) = _set_generatorlevel_values(n, n_g, datapoint_dict, 
        generator_buses, ref_bus)

    # set edge-level values.
    (
        ij_e, ijR_e, Y_re_e, Y_im_e, Yc_ij_im_e, Yc_ijR_im_e, T_mag_e,
        T_ang_e, su_e, vangl_e, vangu_e, Yc_ij_re_e, Yc_ijR_re_e, 
        Y_re_n, Y_im_n, sang_low_e, sang_up_e, suR_e, Y_mag_e, Y_ang_e
    ) = _set_edgelevel_values(n_e, datapoint_dict, Ys_re_n, Ys_im_n)

    # save important grid-level attributes here
    grid_attr = _set_attr_to_dict(
        baseMVA=baseMVA, n=n, n_e=n_e, n_g=n_g, ref_bus=ref_bus, 
        ref_gen_list=ref_gen_list, bustype_n=bustype_n, ij_e=ij_e, 
        ijR_e=ijR_e, gen_bus_g=gen_bus_g
    )

    # save constants here
    const_attr = _set_attr_to_dict(
        vang_low_n=vang_low_n, vang_up_n=vang_up_n
    )

    # save features that you want to standardize or normalize
    orig_attr = _set_attr_to_dict(
        Sd_re_n=Sd_re_n, Sd_im_n=Sd_im_n, Ys_re_n=Ys_re_n, Ys_im_n=Ys_im_n,
        Y_re_n=Y_re_n, Y_im_n=Y_im_n, Sgl_re_n=Sgl_re_n, Sgl_im_n=Sgl_im_n,
        Sgu_re_n=Sgu_re_n, Sgu_im_n=Sgu_im_n, vl_n=vl_n, vu_n=vu_n,
        c0g_n=c0g_n, c1g_n=c1g_n, c2g_n=c2g_n, num_gen_n=num_gen_n,
        ref_node_n=ref_node_n, Y_re_e=Y_re_e, Y_im_e=Y_im_e,
        Yc_ij_re_e=Yc_ij_re_e, Yc_ij_im_e=Yc_ij_im_e, 
        Yc_ijR_re_e=Yc_ijR_re_e, Yc_ijR_im_e=Yc_ijR_im_e, su_e=su_e, 
        suR_e=suR_e, sang_low_e=sang_low_e, sang_up_e=sang_up_e, 
        vangl_e=vangl_e, vangu_e=vangu_e, T_mag_e=T_mag_e, T_ang_e=T_ang_e, 
        Sl_re_g=Sl_re_g, Sl_im_g=Sl_im_g, Su_re_g=Su_re_g, Su_im_g=Su_im_g, 
        c0_g=c0_g, c1_g=c1_g, c2_g=c2_g, mbase_g=mbase_g, Y_mag_e=Y_mag_e, 
        Y_ang_e=Y_ang_e, vm_init_n=vm_init_n, va_init_n=va_init_n
    )

    # transform numpy arrays into tensors
    const_attr = _numpy_to_tensor(const_attr)
    orig_attr = _numpy_to_tensor(orig_attr)
    grid_attr = _numpy_to_tensor(grid_attr)
    
    # set node-level features x
    x_node = _concatenate_features(
        Sd_re_n, Sd_im_n, Ys_re_n, Ys_im_n, Y_re_n, Y_im_n,
        Sgl_re_n, Sgl_im_n, Sgu_re_n, Sgu_im_n, vl_n, vu_n,
        c0g_n, c1g_n, c2g_n, num_gen_n, ref_node_n
    )
    
    # set edge-level features corresponding with edge_index
    x_edge = _concatenate_features(
        Y_re_e, Y_im_e, Yc_ij_re_e, Yc_ij_im_e, Yc_ijR_re_e,
        Yc_ijR_im_e, su_e, vangl_e, vangu_e, T_mag_e, T_ang_e
    )
    
    # set generator-level features x_gen
    x_gen = _concatenate_features(
        Sl_re_g, Sl_im_g, Su_re_g, Su_im_g, c0_g, c1_g, c2_g, mbase_g
    )
    
    # set edge index as concatenation of ij and ijR_e for undirected graph.
    edge_index = torch.from_numpy(np.concatenate((ij_e, ijR_e))).long()
    
    # return a dictionary containing all parsed components of a datapoint
    return {
        "x_node": x_node, 
        "x_edge": x_edge,
        "x_gen": x_gen,
        "edge_index": edge_index,
        "grid_attr": grid_attr,
        "orig_attr": orig_attr, 
        "const_attr": const_attr
    }

def _set_nodelevel_values(
    n, 
    datapoint_dict, 
    load_buses, 
    shunt_buses
):
    """Parse node-level attributes. Noteably, a subset of these values must
    be set at the generator level.
    """

    # initialize values
    Sd_re_n = np.zeros(n, dtype=np_dtype)
    Sd_im_n = np.zeros(n, dtype=np_dtype)
    Ys_re_n = np.zeros(n, dtype=np_dtype)
    Ys_im_n = np.zeros(n, dtype=np_dtype)
    vl_n = np.zeros(n, dtype=np_dtype)
    vu_n = np.zeros(n, dtype=np_dtype)
    ref_node_n = np.ones(n, dtype=np.intc)
    bustype_n = np.zeros(n, dtype=np.intc)
    ref_bus = None
    ref_count = 0

    # initialize values that are not given in data and remain unfilled
    va_init_n = np.zeros(n, dtype=np_dtype)

    # iterate over nodes
    for idx, values in enumerate(datapoint_dict['grid']['nodes']['bus']):
        # set lower and upper voltage magnitude limits of each node
        vl_n[idx] = values[2] * values[0] # minimum * basekV
        vu_n[idx] = values[3] * values[0] # maximum * basekV
        bustype_n[idx] = values[1] # PQ(1), PV(2), reference(3), inactive(4)
        # check if reference bus and save
        if bustype_n[idx] == 3: 
            ref_node_n[idx] = 0
            ref_bus = idx
            ref_count += 1
    
    # iterate over loads
    for idx, values in enumerate(datapoint_dict['grid']['nodes']['load']):
        Sd_re_n[load_buses[idx]] += values[0] # real power demand
        Sd_im_n[load_buses[idx]] += values[1] # reactive power demand
        
    # iterate over shunts
    for idx, values in enumerate(datapoint_dict['grid']['nodes']['shunt']):
        Ys_im_n[shunt_buses[idx]] += values[0] # shunt susceptance
        Ys_re_n[shunt_buses[idx]] += values[1] # shunt conductance
        
    # lower and upper voltage angle limits
    vang_low_n = -np.pi / 2 * np.ones((n), dtype=np_dtype)
    vang_up_n = np.pi / 2 * np.ones((n), dtype=np_dtype)

    return (Sd_re_n, Sd_im_n, Ys_re_n, Ys_im_n, vl_n, vu_n, va_init_n, 
        ref_node_n, bustype_n, ref_bus, ref_count, vang_low_n, vang_up_n)

def _set_generatorlevel_values(
    n, 
    n_g, 
    datapoint_dict, 
    generator_buses, 
    ref_bus
):
    """Parse generator-level attributes and subset of node-leval attributes
    that must be set through iterations on generatir-level.
    """

    # Node-level values that must be filled at generator-level
    vm_init_n = np.ones(n, dtype=np_dtype) # 1 for all load buses.
    Sgl_re_n = np.zeros(n, dtype=np_dtype)
    Sgl_im_n = np.zeros(n, dtype=np_dtype)
    Sgu_re_n = np.zeros(n, dtype=np_dtype)
    Sgu_im_n = np.zeros(n, dtype=np_dtype) 
    c0g_n = np.zeros(n, dtype=np_dtype)
    c1g_n = np.zeros(n, dtype=np_dtype)
    c2g_n = np.zeros(n, dtype=np_dtype)
    num_gen_n = np.zeros(n, dtype=np.intc)

    # initialize generator-level values
    Sl_re_g = np.zeros(n_g, dtype=np_dtype)
    Sl_im_g = np.zeros(n_g, dtype=np_dtype)
    Su_re_g = np.zeros(n_g, dtype=np_dtype)
    Su_im_g = np.zeros(n_g, dtype=np_dtype)
    c0_g = np.zeros(n_g, dtype=np_dtype) # const cost
    c1_g = np.zeros(n_g, dtype=np_dtype) # lin cost
    c2_g = np.zeros(n_g, dtype=np_dtype) # quad cost
    mbase_g = np.zeros(n_g, dtype=np_dtype)
    pg_init_g = np.zeros(n_g, dtype=np_dtype)
    qg_init_g = np.zeros(n_g, dtype=np_dtype)
    gen_bus_g = np.array(generator_buses, dtype=np.intc)
    ref_gen_list = []

    # iterate over generators
    for idx, values in enumerate(
        datapoint_dict['grid']['nodes']['generator']
    ):
        mbase_g[idx] = values[0] # Total MVA base
        pg_init_g[idx] = values[1] # initial real power generation
        Sl_re_g[idx] = values[2] # minimum real power generation
        Su_re_g[idx] = values[3] # maximum rela power generation
        qg_init_g[idx] = values[4] # initial reactive power generation
        Sl_im_g[idx] = values[5] # minimum reactive power generation
        Su_im_g[idx] = values[6] # maximum reactive power generation
        vm_init_n[gen_bus_g[idx]] = values[7] # initial voltage mag PV bus
        c2_g[idx] = values[8] # coefficient pg^2 cost
        c1_g[idx] = values[9] # coefficient pg cost
        c0_g[idx] = values[10] # coefficient constant cost

        # save the generator IDs at slack bus
        if gen_bus_g[idx] == ref_bus:
            ref_gen_list.append(idx)
    
    # transform to numpy array
    ref_gen_list = np.array(ref_gen_list, dtype=np.intc)

    for gen_id, node_id in enumerate(gen_bus_g):
        Sgl_re_n[node_id] += Sl_re_g[gen_id]
        Sgl_im_n[node_id] += Sl_im_g[gen_id]
        Sgu_re_n[node_id] += Su_re_g[gen_id]
        Sgu_im_n[node_id] += Su_im_g[gen_id]
        c0g_n[node_id] += c2_g[gen_id]
        c1g_n[node_id] += c1_g[gen_id]
        c2g_n[node_id] += c0_g[gen_id]
        num_gen_n[node_id] += 1

    return (Sl_re_g, Sl_im_g, Su_re_g, Su_im_g, c0_g, c1_g, c2_g, mbase_g,
        pg_init_g, qg_init_g, gen_bus_g, ref_gen_list, vm_init_n, Sgl_re_n,
        Sgl_im_n, Sgu_re_n, Sgu_im_n, c0g_n, c1g_n, c2g_n, num_gen_n)


def _set_edgelevel_values(
    n_e, 
    datapoint_dict, 
    Ys_re_n, 
    Ys_im_n
):
    """Parse edge-level attributes and a subset of node-level attributes
    that must be retrieved from iterating over edge-level values.
    """

    # set line buses
    ij_line = np.column_stack(
        (
            datapoint_dict['grid']['edges']['ac_line']['senders'], 
            datapoint_dict['grid']['edges']['ac_line']['receivers']
        )
    )

    # set transformer buses
    ij_transformer = np.column_stack(
        (
            datapoint_dict['grid']['edges']['transformer']['senders'], 
            datapoint_dict['grid']['edges']['transformer']['receivers']
        )
    )

    # initialize values
    ij_e = np.vstack((ij_line, ij_transformer), dtype=np.intc)
    Y_re_e = np.zeros(n_e, dtype=np_dtype)
    Y_im_e = np.zeros(n_e, dtype=np_dtype)
    Yc_ij_im_e = np.zeros(n_e, dtype=np_dtype)
    Yc_ijR_im_e = np.zeros(n_e, dtype=np_dtype)
    T_mag_e = np.ones(n_e, dtype=np_dtype)
    T_ang_e = np.zeros(n_e, dtype=np_dtype)
    su_e = np.zeros(n_e, dtype=np_dtype)
    vangl_e = np.zeros(n_e, dtype=np_dtype)
    vangu_e = np.zeros(n_e, dtype=np_dtype)

    # initialize values that are not given in data and remain unfilled
    Yc_ij_re_e = np.zeros(n_e, dtype=np_dtype) 
    Yc_ijR_re_e = np.zeros(n_e, dtype=np_dtype)
    
    # iterate over lines
    for idx, values in enumerate(
        datapoint_dict['grid']['edges']['ac_line']['features']
    ):
        vangl_e[idx] = values[0] # minimm angle difference in radians
        vangu_e[idx] = values[1] # maximum angle difference in radians
        Yc_ij_im_e[idx] = values[2] # line charging suseptance "from" bus
        Yc_ijR_im_e[idx] = values[3] # line charging suseptance "to" bus
        r = values[4] # resistance
        x = values[5] # reactance
        Y_re_e[idx] = r / (r**2 + x**2) # conductance
        Y_im_e[idx] = -x / (r**2 + x**2) # suseptance
        su_e[idx] = values[6] # thermal line limit
    
    # iterate over transformers, continuing with increments on line indices
    for values in datapoint_dict['grid']['edges']['transformer']['features']:
        idx += 1 # increment idx first
        vangl_e[idx] = values[0] # minimum angle difference in radians
        vangu_e[idx] = values[1] # maximum angle difference in radians
        r = values[2] # resistance of transformer
        x = values[3] # reactance of transformer
        Y_re_e[idx] = r / (r**2 + x**2)
        Y_im_e[idx] = -x / (r**2 + x**2)
        su_e[idx] = values[4] # upper power flow limits
        T_mag_e[idx] = values[7] # off nominal turns ratio
        T_ang_e[idx] = values[8] # phase shift angle
        Yc_ij_im_e[idx] = values[9] # line charging suspetance "from" bus
        Yc_ijR_im_e[idx] = values[10] # line charging suseptance "to" bus
        
    # transformation between polar and rectangular form
    Y_mag_e, Y_ang_e = _rectangle_to_polar(Y_re_e, Y_im_e)

    # create edge indices in reverse order
    ijR_e = ij_e[:, [1, 0]]
    
    #creating nodal admittance
    Y_re_n = Ys_re_n.copy()
    Y_im_n = Ys_im_n.copy()
    for branch_k in range(len(ij_e)):
        node_i = ij_e[branch_k, 0]
        node_j = ij_e[branch_k, 1]
        Y_re_n[node_i] += Y_re_e[branch_k]
        Y_re_n[node_j] += Y_re_e[branch_k]
        Y_im_n[node_i] += Y_im_e[branch_k]
        Y_im_n[node_j] += Y_im_e[branch_k]

    # lower and upper voltage angle limits
    sang_low_e = -np.pi / 2 * np.ones((n_e), dtype=np_dtype)
    sang_up_e = np.pi / 2 * np.ones((n_e), dtype=np_dtype)
    
    # set reverse thermal power flow limits as the same as forward limits.
    suR_e = su_e

    return (ij_e, ijR_e, Y_re_e, Y_im_e, Yc_ij_im_e, Yc_ijR_im_e, T_mag_e,
        T_ang_e, su_e, vangl_e, vangu_e, Yc_ij_re_e, Yc_ijR_re_e, 
        Y_re_n, Y_im_n, sang_low_e, sang_up_e, suR_e, Y_mag_e, Y_ang_e
    )

def _set_attr_to_dict(**attributes):
    return {key: value for key, value in attributes.items()}


def _numpy_to_tensor(data_dict):
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            data_dict[key] = torch.from_numpy(value)
        elif isinstance(value, int) or isinstance(value, float):
            data_dict[key] = torch.tensor(value).reshape(1)

    return data_dict

def _concatenate_features(*arrays):
    return torch.stack([torch.from_numpy(arr) for arr in arrays], dim=1)

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


def _rectangle_to_polar(X_re, X_im):
    """Transform passed variable from rectangular to polar form."""
    small_number = 1.e-10
    if isinstance(X_re, torch.Tensor):
        X_mag = torch.sqrt(X_re**2 + X_im**2)
        X_ang = torch.atan(X_im / (X_re + small_number))  # avoids zero division
    elif isinstance(X_re, np.ndarray):
        X_mag = np.sqrt(X_re**2 + X_im**2)
        X_ang = np.arctan(X_im / (X_re + small_number))  # avoids zero division
    else:
        raise TypeError("Inputs must be  torch.Tensors or numpy.ndarrays.")

    return X_mag, X_ang