# AI4Climate: Collection of Machine Learning Tasks and Datasets for Tackling Climate Change


## Overview

1. [Getting started](#getting-started)
2. [Raw data download](docs/downloads.md)
3. [Contributions](docs/contributions.md)


## Getting started

All datasets are provided on Hugging Face Hub. To get started, install the
datasets python package.

```bash
pip install datasets
```

For example, load the "OPFData" dataset with:
```Python
from datasets import load_dataset

dataset = load_dataset("AI4Climate/OPFData")
```

Available datasets are:
- OPFData