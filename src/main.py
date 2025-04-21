"""Main entry point to program for running experiments associated with study.

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
import transformer
import taskdata
import training

PATH_CONFIG = 'config.yml'
PATH_DATA_ROOT = '../donti_group_shared/AI4Climate/processed/'

def main():
    # parse experiment configurations into dictionary
    cfg = configuration.ExperimentConfiguration(PATH_CONFIG)
    
    # load task datasets
    taskdata_dict = taskdata.load_all(cfg, PATH_DATA_ROOT)

    # initialize model
    model = transformer.TransformerBackbone(
        std_vect_dim=cfg.std_vect_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        max_seq_len=cfg.max_seq_len,
        layer_norm_eps=cfg.layer_norm_eps,
        activation=cfg.activation
    ).to(cfg.torch_device)

    # train model
    training.train_model(
        cfg, 
        model,
        taskdata_dict,
        save=False
    )

if __name__ == "__main__":
    main()