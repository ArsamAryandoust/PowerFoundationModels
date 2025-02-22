# OPFData



## ai4climate loading tests

```python
from ai4climate import load_dataset

(
    train,
    valid,
    test
) = load_dataset('OPFData', 'train_small_test_medium')
```



## Hugging Face download tests

```python
from datasets import load_dataset

dataset_opfdata = load_dataset("AI4Climate/OPFData")
dataset_opfdata_raw = load_dataset("AI4Climate/OPFData_raw")
```

```bash
git lfs install
git clone git@hf.co:datasets/AI4Climate/OPFData
git clone git@hf.co:datasets/AI4Climate/OPFData_raw
```

## Raw data download

You can download the raw OPFData datasets using Pytorch Geometric.

```python
!pip install pyg-nightly # Only necessary until PyG 2.6.0 is released.

from torch_geometric.datasets import OPFDataset

# set directory in which you want to download and extract dataset
root = "YOUR_ROOT_DIRECTORY"

# set name of grid topology dataset
case_name = "GRID_TOPOLOGY_NAME" # e.g. pglib_opf_case14_ieee

# choose if you want dataset with n-1 topological perturbations or not.
perturbation = True # set True or False

OPFDataset(
    root=root,
    case_name=case_name,
    topological_perturbations=perturbation
)
```

The full list of available case names is:
- `pglib_opf_case14_ieee`
- `pglib_opf_case30_ieee`
- `pglib_opf_case57_ieee`
- `pglib_opf_case118_ieee`
- `pglib_opf_case500_goc`
- `pglib_opf_case2000_goc`
- `pglib_opf_case4661_sdet`
- `pglib_opf_case6470_rte`
- `pglib_opf_case10000_goc`
- `pglib_opf_case13659_pegase`

Each of these cases can be downloaded with perturbation once set to True and 
once set to False.
