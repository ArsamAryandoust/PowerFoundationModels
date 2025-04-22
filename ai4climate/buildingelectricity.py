"""Loads requested subtask for BuildingElectricity.

"""
import os 
from PIL import Image
from typing import Dict, Any, Tuple, Union, List
import pandas as pd
import gc

import sys


AVAIL_SUBTASKNAMES_LIST = [
    'odd_time_buildings92',
    'odd_space_buildings92',
    'odd_spacetime_buildings92',
    'odd_time_buildings451',
    'odd_space_buildings451',
    'odd_spacetime_buildings451'
]

ZOOM_LEVEL_LIST = [
    'zoom1',
    'zoom2',
    'zoom3'
]

IMAGE_TYPE_LIST = [
    'aspect',
    'ortho',
    'relief',
    'slope'
]

HISTORIC_WINDOW_SIZE = 96
LABEL_WINDOW_SIZE = 96

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

    # exdend local_dir with corresponding profiles
    if subtask_name.endswith('92'):
        local_dir = os.path.join(local_dir, 'profiles_92')
    elif subtask_name.endswith('451'):
        local_dir = os.path.join(local_dir, 'profiles_451')
    else:
        raise VallueError('Check subtask handling. Naming not consistent!')

    # load electric load profiles
    (
        df_loads, 
        building_to_cluster, 
        time_stamps
    ) = _load_electric_load_profiles(local_dir)

    # load building images
    building_images = _load_building_images(local_dir)

    # load cluster images
    cluster_images = _load_cluster_images(local_dir)

    # load meteo data
    meteo_dict = _load_meteo_data(local_dir)

    # pair data
    paired_dataset = _pair_data(
        df_loads,
        meteo_dict,
        building_to_cluster, 
        time_stamps
    )

    del df_loads, building_to_cluster, time_stamps, meteo_dict
    gc.collect()

    # split data into train, val, test data
    train_data, val_data, test_data = _split_data(
        paired_dataset, 
        subtask_name,
        seed
    )

    # Load natural language descriptions
    task_description    = _create_taskdescription()
    subtask_description = _create_subtaskdescription(subtask_name)

    # bundle to training validation and testing data
    subtask_data = {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'building_images': building_images,
        'cluster_images': cluster_images,
        'task_description': task_description,
        'subtask_description': subtask_description
    }

    return subtask_data


def _pair_data(
    df_loads: pd.DataFrame,
    meteo_dict: dict,
    building_to_cluster: dict, 
    time_stamps: pd.DataFrame
) -> dict:
    """
    Pair all data components into single data points and return as dictionary.

    Parameters
    ----------
    df_loads : pd.DataFrame
        Dataframe containing all load profiles in each column, with columns
        being building IDs.
    meteo_dict : dict
        Dictionary of meteorological Dataframe values. One entry per Cluster.
    building_to_cluster : dict
        Mapping from building IDs to cluster IDs.
        
    Returns
    -------
    paired_dataset : dict

    """
    # fill this
    paired_dataset = {}
    datapoint_counter = 0

    # iterate over all columns/building IDs
    for building_id in df_loads.columns:
        # set cluster ID
        cluster_id = building_to_cluster[int(building_id)]

        # set load profile
        load_profile = df_loads[building_id]

        # iterate over loads in window sizes.
        for i in range(
            HISTORIC_WINDOW_SIZE, 
            len(load_profile), 
            HISTORIC_WINDOW_SIZE + LABEL_WINDOW_SIZE
        ):
            # set historic load
            load = load_profile.iloc[i-HISTORIC_WINDOW_SIZE:i].values

            # set future load as label
            label = load_profile.iloc[i:i+LABEL_WINDOW_SIZE].values

            # set time stamp
            timestamp = time_stamps.iloc[i]

            # set meteo data
            meteo_df = meteo_dict[f'cluster_{cluster_id}']
            air_density = meteo_df['air_density'].iloc[
                i-HISTORIC_WINDOW_SIZE:i].values
            cloud_cover = meteo_df['cloud_cover'].iloc[
                i-HISTORIC_WINDOW_SIZE:i].values
            precipitation = meteo_df['precipitation'].iloc[
                i-HISTORIC_WINDOW_SIZE:i].values
            radiation_surface = meteo_df['radiation_surface'].iloc[
                i-HISTORIC_WINDOW_SIZE:i].values
            radiation_toa = meteo_df['radiation_toa'].iloc[
                i-HISTORIC_WINDOW_SIZE:i].values
            snow_mass = meteo_df['snow_mass'].iloc[
                i-HISTORIC_WINDOW_SIZE:i].values
            snowfall = meteo_df['snowfall'].iloc[
                i-HISTORIC_WINDOW_SIZE:i].values
            temperature_celsius = meteo_df['temperature'].iloc[
                i-HISTORIC_WINDOW_SIZE:i].values
            wind_speed = meteo_df['wind_speed'].iloc[
                i-HISTORIC_WINDOW_SIZE:i].values

            data_point = {
                'load': load,
                'air_density': air_density,
                'cloud_cover': cloud_cover,
                'precipitation': precipitation,
                'radiation_surface': radiation_surface,
                'radiation_toa': radiation_toa,
                'snow_mass': snow_mass,
                'snowfall': snowfall,
                'temperature_celsius': temperature_celsius,
                'wind_speed': wind_speed,
                'timestamp': timestamp,
                'building_id': int(building_id),
                'cluster_id': cluster_id,
                'label': label
            }
            paired_dataset.update(
                {datapoint_counter: data_point}
            )
            datapoint_counter += 1    

    return paired_dataset


def _split_data(
    paired_dataset, 
    subtask_name,
    seed
):
    """
    Split train/val/test data according to the chosen subtask.
    
    Parameters
    ----------
    paired_dataset : dict
        Keys are arbitrary indices; values are dictionaries with fields
        such as 'timestamp', 'building_id', 'load', etc.
    subtask_name : str
        Name of the subtask, as defined in AVAIL_SUBTASKNAMES_LIST. One of:
        [
            'odd_time_buildings92',
            'odd_space_buildings92',
            'odd_spacetime_buildings92',
            'odd_time_buildings451',
            'odd_space_buildings451',
            'odd_spacetime_buildings451'
        ]
    seed : int
        Random seed for reproducibility when shuffling building IDs.
        
    Returns
    -------
    train_data : list of dict
    val_data   : list of dict
    test_data  : list of dict

    """
    # Convert the entire paired_dataset to a DataFrame for easy splitting.
    df = pd.DataFrame.from_dict(paired_dataset, orient='index')
    # Ensure timestamps are datetime if not already:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Decide if we are doing time-based, building-based, or both.
    do_time_split   = ('_time_' in subtask_name)
    do_space_split  = ('_space_' in subtask_name)
    do_spacetime    = ('_spacetime_' in subtask_name)

    # set standard splitting ratios as defined by SPLIT_RATIO
    train_ratio = SPLIT_RATIO[0]
    val_ratio = SPLIT_RATIO[1]
    test_ratio = SPLIT_RATIO[2]
    
    ### Helper functions:
    def time_based_split(df_in):
        """
        Split df_in purely by timestamp in ascending order, according
        to train/val/test ratios. All building IDs remain in each split.

        """
        df_sorted = df_in.sort_values(by='timestamp').reset_index(drop=True)
        n_total   = len(df_sorted)
        n_train   = int(train_ratio * n_total)
        n_val     = int(val_ratio   * n_total)
        # The remainder automatically goes to test
        n_test    = n_total - n_train - n_val
        
        df_train = df_sorted.iloc[:n_train]
        df_val   = df_sorted.iloc[n_train:(n_train + n_val)]
        df_test  = df_sorted.iloc[(n_train + n_val):]
        
        return df_train, df_val, df_test

    def building_based_split(df_in):
        """
        Split df_in purely by building ID. Each building appears in exactly
        one of train/val/test sets, with no overlap of IDs across splits.

        """
        building_ids = df_in['building_id'].unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(building_ids)
        
        n_bldg       = len(building_ids)
        n_train_bldg = int(train_ratio * n_bldg)
        n_val_bldg   = int(val_ratio   * n_bldg)
        # remainder to test
        n_test_bldg  = n_bldg - n_train_bldg - n_val_bldg
        
        bldg_train = building_ids[:n_train_bldg]
        bldg_val   = building_ids[n_train_bldg:(n_train_bldg + n_val_bldg)]
        bldg_test  = building_ids[(n_train_bldg + n_val_bldg):]

        df_train = df_in[df_in['building_id'].isin(bldg_train)]
        df_val   = df_in[df_in['building_id'].isin(bldg_val)]
        df_test  = df_in[df_in['building_id'].isin(bldg_test)]
        
        return df_train, df_val, df_test

    # 1) "odd_time_*" => all buildings in each split, but time-based split
    if (do_time_split) and (not do_space_split):
        df_train, df_val, df_test = time_based_split(df)

    # 2) "odd_space_*" => building-based partition
    elif (do_space_split) and (not do_time_split):
        df_train, df_val, df_test = building_based_split(df)
        
    # 3) "odd_spacetime_*" => disjoint building IDs AND disjoint timestamps
    elif do_spacetime:
        # Randomly pick set of building IDs for test (disjoint from train/val).
        # Among remaining building IDs, do a time-based split for train and val
        # Test set also uses a disjoint time window among the test building IDs
        
        building_ids = df['building_id'].unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(building_ids)
        
        # Let test portion of building IDs be test_ratio
        n_bldg            = len(building_ids)
        n_test_bldg       = int(test_ratio * n_bldg)
        test_bldg         = building_ids[:n_test_bldg]
        trainval_bldg     = building_ids[n_test_bldg:]
        
        # From the building subset used for train+val, do a time-based split:
        df_trainval = df[df['building_id'].isin(trainval_bldg)]
        df_trainval_sorted = df_trainval.sort_values(
            by='timestamp').reset_index(drop=True)

        # split remainders
        trainval_ratio = train_ratio + val_ratio # for rescaled ratios
        n_trainval = len(df_trainval_sorted)
        n_train    = int(train_ratio / trainval_ratio * n_trainval)
        
        df_train = df_trainval_sorted.iloc[:n_train]
        df_val   = df_trainval_sorted.iloc[n_train:]
        
        # set test building IDs and sort by time stamp for splitting off.
        df_test_bldg = df[df['building_id'].isin(test_bldg)]
        df_test_bldg_sorted = df_test_bldg.sort_values(by='timestamp').reset_index(drop=True)

        # split off time and building
        n_test_bldg_data  = len(df_test_bldg_sorted)
        n_test_time = int( test_ratio * n_test_bldg_data )
        df_test     = df_test_bldg_sorted.iloc[-n_test_time:]
        # (In that scenario, the building IDs and timestamps in test do not appear in train/val.)

    else:
        raise ValueError(f"Unexpected subtask pattern: {subtask_name}")
    
    # Convert dataframes back to lists of dictionaries
    train_data = df_train.to_dict(orient='records')
    val_data   = df_val.to_dict(orient='records')
    test_data  = df_test.to_dict(orient='records')
    
    return train_data, val_data, test_data



def _load_meteo_data(local_dir):
    """
    Load meteorological time series data.

    Parameters
    ----------
    local_dir : str
        path to profile subset data directory root.


    Returns
    ----------
    meteo_dict : dict of pd.DataFrame

    """
    # fill this
    meteo_dict = {}

    # path 
    path_meteo = os.path.join(local_dir, 'meteo_data')

    # read directory
    meteo_filename_list = os.listdir(path_meteo)

    # iterate over all filenames
    for filename in meteo_filename_list:
        # set path to file
        path_load = os.path.join(path_meteo, filename)

        # load
        meteo_file = pd.read_csv(path_load)

        # get file name key for dict
        filekey = filename.replace('.csv', '')

        # save file
        meteo_dict[filekey] = meteo_file
    
    return meteo_dict


def _load_electric_load_profiles(local_dir):
    """
    Load electric load profiles as DataFrame.

    Parameters
    ----------
    local_dir : str
        path to profile subset data directory root.


    Returns
    ----------
    df_loads : pd.DataFrame
    building_to_cluster : dict of int
    time_stamps : pd.Series

    """
    # set path
    path_load = os.path.join(local_dir, 'load_profiles.csv')

    # load csv
    df_loads = pd.read_csv(path_load)

    # First row is the data, first row index is probably 0
    time_stamps = df_loads.iloc[1:, 0]
    cluster_ids = df_loads.iloc[0, 1:] #skip first column (label "cluster ID")
    building_ids = df_loads.columns[1:] #skip first column (label "building ID")

    # drop cluster ID row
    df_loads.drop(labels=1, axis='index', inplace=True)

    # drop building ID column
    df_loads.drop(columns='building ID', inplace=True)

    # Create the dictionary
    building_to_cluster = dict(
        zip(building_ids.astype(int), 
        cluster_ids.astype(int))
    )

    return df_loads, building_to_cluster, time_stamps

    

def _load_building_images(local_dir):
    """
    Load aerial images of buildings. Use padded images.

    Parameters
    ----------
    local_dir : str
        path to profile subset data directory root.


    Returns
    ----------
    building_image_dict : dict of images

    """
    # fill this dictionary
    building_image_dict = {}

    # set paths and load. Use padded images.
    path_images = os.path.join(local_dir, 'building_images', 'padded')

    # list all files
    image_file_list = os.listdir(path_images)

    # iterate over all filenames.
    for filename in image_file_list:
        # check if png
        if not filename.endswith('.png'):
            continue

        # set path
        path_load = os.path.join(path_images, filename)

        # load file
        image = Image.open(path_load).convert('RGB')

        # get building ID
        building_id = filename.split('_')[1].replace('.png', '')

        # save image
        building_image_dict[building_id] = image

    return building_image_dict


def _load_cluster_images(local_dir):
    """
    Load aerial images of clusters.

    Parameters
    ----------
    local_dir : str
        path to profile subset data directory root.


    Returns
    ----------
    cluster_image_dict : dict of images

    """
    # fill this dictionary
    cluster_image_dict = {}

    # set path
    path_cluster_images = os.path.join(local_dir, 'cluster_images')

    # iterate over all zoom levels
    for zoom_level in ZOOM_LEVEL_LIST:
        
        # fill with image types
        image_type_dict = {}

        # iterate over all types
        for image_type in IMAGE_TYPE_LIST:

            # fill with cluster images
            cluster_image_dict = {}
            
            # set path
            path_images_dir = os.path.join(path_cluster_images, zoom_level,
                image_type)

            # read directory
            image_file_list = os.listdir(path_images_dir)

            # iterate over directory
            for filename in image_file_list:
                if not filename.endswith('.png'):
                    continue
                
                # load path
                path_load = os.path.join(path_images_dir, filename) 

                # load file
                image = Image.open(path_load).convert('RGB')

                # set cluster id
                key_imagename = filename.replace('.png', '')

                # fill cluster image dictionary
                cluster_image_dict[key_imagename] = image

            # fill image type dictionary
            image_type_dict[image_type] = cluster_image_dict

        # save for zoom level
        cluster_image_dict[zoom_level] = image_type_dict

    return cluster_image_dict

def _create_taskdescription():
    """Contains natural language description of task. Placeholder."""

    task_description = """
    Given the aerial image of a building and the meteorological conditions in 
    the region of that building for a past time window of 24 hours, the goal in 
    BuildingElectricity is to predict the electric load profile of single 
    buildings for a future time window of 24 hours.

    This is a short-term spatial demand forecasting challenge, where accurate 
    can significantly support the planning and dispatch of distributed renewable 
    energy sources, such as rooftop photovoltaics and micro-wind turbines, the 
    allocation and sizing of storage capacities, and the coordination of flexible 
    loads through demand response programs.

    The primary underlying pattern governing this task is the circadian rhythm 
    of human behavior, a weak but consistent physical law that shapes daily 
    electricity demand.

    We distinguish six sub-tasks, a first set containing data points from 92 
    different buildings and a second set containing data points from 451 
    different buildings:

    odd_time_buildings92: Predict electric load profiles from 92 buildings, 
    with test time stamps not present in training data.

    odd_space_buildings92: Predict electric load profiles from 92 buildings,
    with test buildings not present in training data.

    odd_spacetime_buildings92: Predict electric load profiles from 92 buildings, 
    with test time stamps and buildings not present in training data.

    odd_time_buildings451: Predict electric load profiles from 451 buildings,
    with test time stamps not present in training data.

    odd_space_buildings451: Predict electric load profiles from 451 buildings, 
    with test buildings not present in training data.

    odd_spacetime_buildings451: Predict electric load profiles from 451
    buildings, with test buildings and time stamps not present in training data.
    """

    return task_description

def _create_subtaskdescription(subtask_name: str):
    """Contains natural language description of subtask. Placeholder."""

    subtask_description = f"""
    Here, we are solving instances of the {subtask_name} subtask.
    """.format(subtask_name)
    
    return subtask_description

    