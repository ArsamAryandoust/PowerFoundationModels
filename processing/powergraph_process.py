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

datapoints_per_file = 100

def main():
    """ """
    # create a list of paths to datasets
    path_list = [
        #path_texas,
        #path_uk,
        path_ieee24,
        path_ieee39,
        path_ieee118
    ]

    # create a list of corresponding data names
    name_list = [
        #'texas',
        #'uk',
        'ieee24',
        'ieee39',
        'ieee118'
    ]

    # iterate over paths
    for id_case, data_path in enumerate(path_list):
        
        #####################################
        # Set up path and folders
        #####################################

        # case name
        casename = name_list[id_case]

        # path case
        path_target = os.path.join(path_root_target, casename)
        
        # create directories if not existent
        os.makedirs(path_target, exist_ok=True)


        #####################################
        # Load and extract data
        #####################################

        # load data 
        Bf = mat73.loadmat(os.path.join(data_path, 'Bf.mat'))
        blist = mat73.loadmat(os.path.join(data_path, 'blist.mat'))
        Ef_nc = mat73.loadmat(os.path.join(data_path, 'Ef_nc.mat'))
        Ef = mat73.loadmat(os.path.join(data_path, 'Ef.mat'))
        exp = mat73.loadmat(os.path.join(data_path, 'exp.mat'))
        of_bi = mat73.loadmat(os.path.join(data_path, 'of_bi.mat'))
        of_mc = mat73.loadmat(os.path.join(data_path, 'of_mc.mat'))
        of_reg = mat73.loadmat(os.path.join(data_path, 'of_reg.mat'))

        # exctract data point lists from dictionary
        Bf = Bf['B_f_tot']
        blist = blist['bList']
        Ef_nc = Ef_nc['E_f_kenza']
        Ef = Ef['E_f_post']
        exp = exp['explainations']
        of_bi = of_bi['output_features']
        of_mc = of_mc['category']
        of_reg = of_reg['dns_MW']


        #####################################
        # Create data points as dictionaries
        #####################################
        n_datapoints = len(Bf)

        # iterate over data points
        filename_counter = 1
        for i in range(0, n_datapoints, datapoints_per_file):
            data_dict = {}
            for j in range(i, i+datapoints_per_file):

                # create node_features
                node_features = {
                    1: Bf[j][0][:, 0].tolist(),
                    2: Bf[j][0][:, 1].tolist(),
                    3: Bf[j][0][:, 2].tolist()
                }

                # create edges_features
                edges_features = {
                    1: Ef_nc[j][0][:, 0].tolist(),
                    2: Ef_nc[j][0][:, 1].tolist(),
                    3: Ef_nc[j][0][:, 2].tolist(),
                    4: Ef_nc[j][0][:, 3].tolist(),
                    5: Ef[j][0][:, 0].tolist(),
                    6: Ef[j][0][:, 1].tolist(),
                    7: Ef[j][0][:, 2].tolist(),
                    8: Ef[j][0][:, 3].tolist()
                }

                # create labels_dict
                labels = {
                    1: of_bi[j][0].item(),
                    2: of_mc[j][0].tolist(),
                    3: of_reg[j],
                    4: exp[j][0] if exp[j][0] is None else exp[j][0].tolist()
                }

                edge_index = {
                    1: blist[:, 0].tolist(),
                    2: blist[:, 1].tolist()
                }

                # add data point to data dictionary
                data_dict[j] = {
                    'nodes': node_features,
                    'edges': edges_features,
                    'edge_index': edge_index,
                    'labels': labels
                }
            
            # save data dictionary as json file
            filename = f'merged_{filename_counter}.json'
            path_save = os.path.join(path_target, filename)
            with open(path_save, 'w') as json_outfile:
                json.dump(data_dict, json_outfile)
            
            # increment filename counter
            filename_counter += 1


if __name__ == '__main__':
    main()
    print("Successfully processed Power Graph data.")