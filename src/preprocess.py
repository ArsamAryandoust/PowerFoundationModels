"""Contains data standardization modules.

"""
import torch

def standardize(
    cfg: "congifuration.ExperimentConfiguration",
    taskname: str, 
    taskdata: dict
):
    """
    Standardize datasets for training and validation loop with MultiSenseNet.
    
    """
    if taskname == 'OPFData':
        taskdata = _prep_opfdata(cfg, taskdata)
    elif taskname == 'PowerGraph':
        taskdata = _prep_powergraph(cfg, taskdata)
    elif taskname == 'SolarCube':
        taskdata = _prep_solarcube(cfg, taskdata)
    elif taskname == 'BuildingElectricity':
        taskdata = _prep_buildingelectricity(cfg, taskdata)
    elif taskname == 'WindFarm':
        taskdata = _prep_windfarm(cfg, taskdata)

    return taskdata

def _prep_opfdata(
    cfg: "congifuration.ExperimentConfiguration",
    taskdata: dict
):
    """ """

    return taskdata

def _prep_powergraph(
    cfg: "congifuration.ExperimentConfiguration",
    taskdata: dict
):
    """ """

    return taskdata

def _prep_solarcube(
    cfg: "congifuration.ExperimentConfiguration",
    taskdata: dict
):
    """ """

    return taskdata

def _prep_buildingelectricity(
    cfg: "congifuration.ExperimentConfiguration",
    taskdata: dict
):
    """ """

    return taskdata


def _prep_windfarm(
    cfg: "congifuration.ExperimentConfiguration",
    taskdata: dict
) -> dict:
    """
    Process WindFarm data as requested by ai4climate package.
    
    """
        

    return taskdata

