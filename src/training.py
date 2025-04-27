"""Contains training, validation and testing procedure.

"""
import torch

import dataprocessing
import multisensenet


def train_model(
    cfg: "congifuration.ExperimentConfiguration",
    taskdata_dict: dict,
    update_models: bool
):
    """ 
    Train backbone network and encoder parts.

    Parameters
    ----------
    cfg : congifuration.ExperimentConfiguration
        Class object that bundles configurations for experiment.
    taskdata_dict : dict
        Task data loaded as requested by user.
    update_models : bool 
        Whether or not to save updates to model weights.

    """
    # iterate over all tasks in taskdata_dict
    for taskname, taskdata in taskdata_dict.items():
        if taskdata is None:
            continue
        
        # prepare data
        taskdata = dataprocessing.standardize(cfg, taskname, taskdata)

        # create model instance
        model = multisensenet.make_model(cfg, taskname)
        
        # enter epochs
        for epoch in range(cfg.epochs):
            with torch.no_grad():
                # validate
                _exec_epoch(cfg, taskdata, model, mode='val')
                # test
                _exec_epoch(cfg, taskdata, model, mode='test')

            # training
            _exec_epoch(cfg, taskdata, model, mode='train')


def _exec_epoch(
    cfg: "congifuration.ExperimentConfiguration",
    taskdata: dict,
    model: "multisensenet.TransformerBackbone",
    mode: str,
):
    """ """
    # set model according to mode
    model.train() if mode == 'train' else model.eval() 

    # get data
    dummy_input = taskdata['dummy_input']
    padding_mask = taskdata['padding_mask']
    
    # make predictions
    output = model(dummy_input, src_key_padding_mask=padding_mask)
    print("Output shape:", output.shape)  # (2, 128, 1024)


    
    



