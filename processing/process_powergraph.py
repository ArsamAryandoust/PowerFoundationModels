"""Contains multi-processing functions that pre-process datasets in parallel.

Main entry point to script-wise data pre-processing. Contains function that can
be called individually


Example:
--------
    
    $ python process_powergraph.py
    
"""
import os
import mat73

# set up base paths
path_root_target = '../../donti_group_shared/AI4Climate/processed/PowerGraph'
path_root_source = '../../donti_group_shared/AI4Climate/raw/PowerGraph_raw'
path_texas = os.path.join(path_root_source, 'raw')
path_uk = os.path.join(path_root_source, 'dataset_cascades/uk/uk/raw')
path_ieee24 = os.path.join(path_root_source, 'dataset_cascades/ieee24/ieee24/raw')
path_ieee39 = os.path.join(path_root_source, 'dataset_cascades/ieee39/ieee39/raw')
path_ieee118 = os.path.join(path_root_source, 'dataset_cascades/ieee118/ieee118/raw')


def main():
    """ """
    # create a list of paths to datasets
    path_list = [
        path_texas,
        path_uk,
        path_ieee24,
        path_ieee39,
        path_ieee118
    ]

    # iterate over paths
    for data_path in path_list:
        # list files
        file_list = os.listdir(path_texas)

        print(file_list)





if __name__ == '__main__':
    main()