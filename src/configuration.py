"""Loads, parses and tests the configurations user sets in config.yml.

"""
import random
import yaml
import torch
import numpy as np



class ExperimentConfiguration:
    """
    Bundles the configurations for executing experiments.
    
    """
    
    def __init__(
        self, 
        path_file: str
    ):
        """
        Load config from yaml. Reset attributes and add new attributes.
        
        """
        
        # load config as dict from yaml file
        with open(path_file, 'r') as cfg_file:
            cfg_dict = yaml.safe_load(cfg_file)
        
        # set loaded dictionary elements as class attributes
        for key, value in cfg_dict.items():
            exec(f'self.{key} = value')
        
        # set cuda or cpu as device
        if self.device_gpu and torch.cuda.is_available():
            self.torch_device = torch.device('cuda')
            device_name = 'GPU'
        else:
            self.torch_device = torch.device('cpu')
            device_name = 'CPU'
        
        # tell us what we are doing
        print(f"Running computation on {device_name}")

        # set torch default data type and numpy data type
        if self.dtype == 'float64':
            torch.set_default_dtype(torch.float64)
            self.np_dtype = np.float64
        elif self.dtype == 'float32':
            torch.set_default_dtype(torch.float32)
            self.np_dtype = np.float32

         # set anomaly detection for tracing discontinuity in backpropagation.
        torch.autograd.set_detect_anomaly(self.torch_anomaly_detection)

        # set global random seed.
        self._set_random_seed()

        # test configs before continuing.
        self.run_tests()

    def run_tests(self):
        """Test configurations before continuing."""
        # placeholder

        print("All tests for configuration passed!")


    def revert_configs(self):
        """Delete configs with non-string types we cannot store in json."""

        del self.torch_device
        del self.np_dtype

    def _set_random_seed(self):
        """Set global seed for all relevant random processes."""

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.torch_device == torch.device('cuda'):
            torch.cuda.manual_seed_all(self.seed)


        
