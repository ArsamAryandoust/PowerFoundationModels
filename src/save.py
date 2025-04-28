"""Save results.

"""
import json
import os

def all_results(
    path_results: str,
    cfg: "configuration.ExperimentConfiguration"
):
    """
    Save configuration and results

    Parameters
    ----------
    path_results : str
        Path to root of result saving repo.
    cfg : configuration.ExperimentConfiguration
        Main configuration file for experiments we have run.
    
    """
    ### Create repository structure
    path_experiments = os.path.join(path_results, cfg.exp_name)
    os.makedirs(path_experiments, exist_ok=True)

    ### Save configuration
    # revert non-numeric and non-string attributes
    cfg.revert_configs()
    # full saving path
    path_save = os.path.join(path_experiments, 'config.json')
    # save as json
    with open(path_save, 'w') as config_json:
        json.dump(cfg.__dict__, config_json)
    print(f"Configuration file saved under {path_save}")

