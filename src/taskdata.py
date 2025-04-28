"""Prepares task data.

"""
import ai4climate
from ai4climate import load

def load_all(
    cfg: "configuration.ExperimentConfiguration",
    path_data_root: str
) -> dict:
    """Load all requested task datsets.
    
    Parameters
    -----------
    cfg : configuration.ExperimentConfiguration
        Class object bundling all experiment configurations.
    path_data_root : str
        Path to root directory of stored data.

    Returns
    -----------
    dict
        Dictionary containing all requested task datasets.

    """
    # overwrite where requested
    opfdata_taskdata = None
    powergraph_taskdata = None
    solarcube_taskdata = None
    buildingelectricity_taskdata = None

    if cfg.opfdata:
        opfdata_taskdata = load.load_task(
            'OPFData', 
            cfg.opfdata_subtask,
            path_data_root,
            data_frac = cfg.data_frac,
            train_frac = cfg.train_frac
        )

    if cfg.powergraph:
        powergraph_taskdata = load.load_task(
            'PowerGraph', 
            cfg.powergraph_subtask,
            path_data_root,
            data_frac = cfg.data_frac,
            train_frac = cfg.train_frac
        )

    if cfg.solarcube:
        solarcube_taskdata = load.load_task(
            'SolarCube', 
            cfg.solarcube_subtask,
            path_data_root,
            data_frac = cfg.data_frac,
            train_frac = cfg.train_frac
        )

    if cfg.buildingelectricity:
        buildingelectricity_taskdata = load.load_task(
            'BuildingElectricity', 
            cfg.buildingelectricity_subtask,
            path_data_root,
            data_frac = cfg.data_frac,
            train_frac = cfg.train_frac
        )

    if cfg.windfarm:
        windfarm_taskdata = load.load_task(
            'WindFarm',
            cfg.windfarm_subtask,
            path_data_root,
            data_frac = cfg.data_frac,
            train_frac = cfg.train_frac
        )
        
    return {
        'opfdata_taskdata': opfdata_taskdata,
        'powergraph_taskdata': powergraph_taskdata,
        'solarcube_taskdata': solarcube_taskdata,
        'buildingelectricity_taskdata': buildingelectricity_taskdata,
        'windfarm_taskdata': windfarm_taskdata
    }