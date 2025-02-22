# AI4Climate: Collection of Machine Learning Tasks and Datasets for Tackling Climate Change


## Overview

1. [Getting started](#getting-started)
2. [Available datasets](#list-of-available-datasets)
2. [Downloads](docs/downloads.md)
3. [Contributions](docs/contributions.md)


## Getting started

All datasets are provided on Hugging Face Hub and ready to be downloaded and 
parsed into our standardized data format with training, validation and testing 
splits using our `ai4climate` Python package. To get started, install the
ai4climate Python package, if you have not already:

```bash
pip install ai4climate
```

For example, load the "train_small_test_medium" task from the "OPFData" dataset:
```Python
from ai4climate import load_dataset

dataset = load_dataset('OPFData', 'train_small_test_medium')
```

## List of available datasets

1. [OPFData](docs/opfdata.md)

