"""Main entry point to program for running experiments.

Example usage
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
import save

PATH_CONFIG = 'config.yml'
PATH_DATA_ROOT = '../donti_group_shared/AI4Climate/processed/'
PATH_RESULTS = 'results/'

def main():
    # parse experiment configurations into dictionary
    cfg = configuration.ExperimentConfiguration(PATH_CONFIG)
    
    # load task datasets
    taskdata_dict = taskdata.load_all(cfg, PATH_DATA_ROOT)

    # train model
    training.train_model(
        cfg,
        taskdata_dict,
        update=True
    )

    # save configuration and results
    save.all_results(PATH_RESULTS, cfg)


if __name__ == "__main__":
    main()