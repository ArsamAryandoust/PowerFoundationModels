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

PATH_CONFIG = 'config.yml'
PATH_DATA_ROOT = '../donti_group_shared/AI4Climate/processed/'

def main():
    # parse experiment configurations into dictionary
    cfg = configuration.ExperimentConfiguration(PATH_CONFIG)
    
    # load task datasets
    (
        opfdata_taskdata,
        powergraph_taskdata,
        solarcube_taskdata,
        buildingelectricity_taskdata
    ) = taskdata.load_all(cfg, PATH_DATA_ROOT)

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

    # Placeholder tests for model
    BATCH = 2
    SEQ_LEN = 128

    dummy_input = torch.randn(
        BATCH, SEQ_LEN, cfg.std_vect_dim, device=cfg.torch_device
    )
    padding_mask = torch.zeros(
        BATCH, SEQ_LEN, dtype=torch.bool, device=cfg.torch_device
    )

    with torch.no_grad():
        output = model(dummy_input, src_key_padding_mask=padding_mask)

    print("Output shape:", output.shape)  # (2, 128, 1024)


if __name__ == "__main__":
    main()