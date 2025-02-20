"""Load standard train-val-test dataset splits."""
from sgs_utils.constants import Benchmark

import os
import math
import json
import random
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed


def load(
    root: str, 
    dataset_name: str, 
    data_frac: float = 1,
    train_frac: float = 1,
    seed: int = None,
    max_workers: int = 2
):
    """ """
    print(f"Loading benchmark data for {dataset_name}")
    
    

