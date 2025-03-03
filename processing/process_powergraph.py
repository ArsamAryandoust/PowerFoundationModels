"""Contains multi-processing functions that pre-process datasets in parallel.

Main entry point to script-wise data pre-processing. Contains function that can
be called individually


Example:
--------
    
    $ python process_powergraph.py
    
"""
import os
import json
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

    # create a list of corresponding data names
    name_list = [
        'texas',
        'uk',
        'ieee24',
        'ieee39',
        'ieee118'
    ]

    # iterate over paths
    for id_case, data_path in enumerate(path_list):

        # case name
        casename = name_list[id_case]

        # path case
        path_target = os.path.join(path_root_target, casename)
        
        # create directories if not existent
        os.makedirs(path_target, exist_ok=True)

        # list files
        file_list = os.listdir(path_texas)
    
        # iterate over file list
        for filename in file_list:

            # set path to file
            path_file = os.path.join(data_path, filename)

            # load .mat file
            file = mat73.loadmat(path_file)
            
            # empty json file
            json_dict = {}

            # process files
            if filename == 'Bf.mat':

                # set entry list
                entry_list = file['B_f_tot']

                # iterate overa all entries
                for entry_id, entry in enumerate(entry_list):

                    # add entry to json dictionary
                    json_dict[entry_id] = entry[0].tolist()

                # 
                filename_save = 'Bf.json'

                # create path for saving results
                path_save = os.path.join(path_target, filename_save)

                with open(path_save, "w") as f:
                    json.dump(json_dict, f)


if __name__ == '__main__':
    main()