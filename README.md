# AI4Climate: Collection of Machine Learning Tasks and Datasets for Tackling Climate Change

## Overview

1. [Getting started](#getting-started)
2. [Available datasets](#list-of-available-datasets)
3. [Contributions](docs/contributions.md)


## Getting started

All datasets are provided on Hugging Face Hub and ready to be downloaded and 
parsed into our standardized data format with training, validation and testing 
splits using our `ai4climate` Python package. Install package:

```bash
pip install ai4climate
```

For example, load the "train_small_test_medium" task from the "OPFData" dataset:
```Python
from ai4climate import load

dataset = load(
    task_name='OPFData', 
    subtask_name='train_small_test_medium',
    root_path='~/AI4Climate/'
)
```

Run main experiments as specified in config.yml with:
```bash
python src/main.py
```

## List of available datasets

1. [OPFData](docs/opfdata.md)
2. [PowerGraph](docs/powergraph.md)
3. [SolarCube](docs/solarcube.md)
4. [BuildingElectricity](docs/buildingelectricity.md)
5. [WindFarm](docs/windfarm.md)