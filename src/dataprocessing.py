"""Contains data standardization modules.

"""
import torch

def standardize(
    cfg: "congifuration.ExperimentConfiguration",
    taskname: str, 
    taskdata: dict
):
    """ """

    if taskname == 'OPFData':
        taskdata = prep_opfdata(taskdata)
    elif taskname == 'PowerGraph':
        taskdata = prep_powergraph(taskdata)
    elif taskname == 'SolarCube':
        taskdata = prep_solarcube(taskdata)
    elif taskname == 'BuildingElectricity':
        taskdata = prep_buildingelectricity(taskdata)


    # placeholder format
    BATCH = 2
    SEQ_LEN = 128

    dummy_input = torch.randn(
        BATCH, SEQ_LEN, cfg.std_vect_dim, device=cfg.torch_device
    )
    padding_mask = torch.zeros(
        BATCH, SEQ_LEN, dtype=torch.bool, device=cfg.torch_device
    )

    taskdata = {
        'dummy_input': dummy_input,
        'padding_mask': padding_mask
    }

    return taskdata

def prep_opfdata(
    taskdata: dict
):
    """ """

    return taskdata

def prep_powergraph(
    taskdata: dict
):
    """ """

    return taskdata

def prep_solarcube(
    taskdata: dict
):
    """ """

    return taskdata

def prep_buildingelectricity(
    taskdata: dict
):
    """ """

    return taskdata

