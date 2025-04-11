"""Loads requested subtask for BuildingElectricity.

"""
import os 
from PIL import Image
from typing import Dict, Any, Tuple, Union, List
import pandas as pd


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
    df_loads, building_to_cluster, time_stamps = load_electric_load_profiles(local_dir)

    # load building images
    building_image_dict = load_building_images(local_dir)

    # load cluster images
    cluster_image_dict = load_cluster_images(local_dir)

    # load meteo data
    meteo_dict = load_meteo_data(local_dir)

    # pair data

    
    # split data into train, val, test data
    train_data, val_data, test_data = 0, 0, 0

    # bundle to training validation and testing data
    subtask_data = {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data
    }

    return subtask_data


def load_meteo_data(local_dir: str):
    """
    Load meteorological time series data.

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


def load_electric_load_profiles(local_dir: str):
    """
    Load electric load profiles as DataFrame.

    """
    # set path
    path_load = os.path.join(local_dir, 'load_profiles.csv')

    # load csv
    df_loads = pd.read_csv(path_load)

    # First row is the data, first row index is probably 0
    time_stamps = df_loads.iloc[1:, 0]
    cluster_ids = df_loads.iloc[0, 1:]  # skip the first column (label "cluster ID")
    building_ids = df_loads.columns[1:]  # skip the first column (label "building ID")

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

    

def load_building_images(local_dir):
    """
    Load aerial images of buildings. Use padded images.

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


def load_cluster_images(local_dir):
    """
    Load aerial images of clusters.

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