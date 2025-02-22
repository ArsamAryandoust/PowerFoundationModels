# Downloads

We host both the raw and processed datasets of the AI4Climate collection on
Hugging Face Hub. Users can download both of these using either Hugging Face's
python package called `datasets` or Git's large file system (LFS) hierarchy.


## Using the Hugging Face `datasets` Library

To load the datasets with Python first install the `datasets`package:
```bash
pip install datasets
```

Then, use the following code snippets in your Python script or interactive 
environment to load <dataset_name> or <dataset_name_raw>:
```Python
from datasets import load_dataset

dataset_x = load_dataset("AI4Climate/<dataset_name>")
dataset_x_raw = load_dataset("AI4Climate/<dataset_name_raw>")
```

For example, download procesed and raw "OPFData" dataset files with
```Python
from datasets import load_dataset

dataset_opfdata = load_dataset("AI4Climate/OPFData")
dataset_opfdata_raw = load_dataset("AI4Climate/OPFData_raw")
```


## Using Git LFS

First initialize Git LFS:
```bash
git lfs install
```

Then, use 
```bash
git clone git@hf.co:datasets/<path_to_dataset_ID>
```

For example, download procesed and raw "OPFData" dataset files with:
```bash
git clone git@hf.co:datasets/AI4Climate/OPFData
git clone git@hf.co:datasets/AI4Climate/OPFData_raw
```

