"""Main entry point to program for running experiments.

Example usage:
--------------
$ python src/main.py

"""
import os
import sys 

import torch

sys.path.append('.')
sys.path.append('ai4climate')

import configuration
import taskdata
import training

PATH_CONFIG = 'config.yml'
PATH_DATA_ROOT = '../donti_group_shared/AI4Climate/processed/'

def main():
    # parse experiment configurations into dictionary
    cfg = configuration.ExperimentConfiguration(PATH_CONFIG)
    
    # load task datasets
    taskdata_dict = taskdata.load_all(cfg, PATH_DATA_ROOT)

    # train model
    training.train_model(
        cfg,
        taskdata_dict,
        update_models=True
    )

if __name__ == "__main__":
    main()