"""Prepares task data.

"""
import ai4climate
from ai4climate import load

def load_all(
    cfg: "configuration.ExperimentConfiguration",
    path_data_root: str
):
    """Load all requested task datsets.
    
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
            data_frac = cfg.data_frac
        )

    if cfg.powergraph:
        powergraph_taskdata = load.load_task(
            'PowerGraph', 
            cfg.powergraph_subtask,
            path_data_root,
            data_frac = cfg.data_frac
        )

    if cfg.solarcube:
        solarcube_taskdata = load.load_task(
            'SolarCube', 
            cfg.solarcube_subtask,
            path_data_root,
            data_frac = cfg.data_frac
        )

    if cfg.buildingelectricity:
        buildingelectricity_taskdata = load.load_task(
            'BuildingElectricity', 
            cfg.buildingelectricity_subtask,
            path_data_root,
            data_frac = cfg.data_frac
        )

