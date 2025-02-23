# Downloads

We host both the raw and processed datasets of the AI4Climate collection on
Hugging Face Hub. Users can progamatically download both of these using one of
the following methods



## Using Git LFS

Download <dataset_name>:
```bash
git lfs install
git clone git@hf.co:datasets/AI4Climate/<dataset_name>
```

For example:
```bash
git clone git@hf.co:datasets/AI4Climate/OPFData
```


## Using `huggingface-cli` library

Download <dataset_name>:
```bash
huggingface-cli download AI4Climate/<dataset_name> --repo-type dataset
```


## Using `huggingface_hub` library

To load the datasets with Python first install the `huggingface_hub`package:
```bash
pip install huggingface_hub
```

Then, use the following code snippets in your Python script or interactive 
environment to load <dataset_name> or <dataset_name_raw>:
```Python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="AI4Climate/<dataset_name>")
snapshot_download(repo_id="AI4Climate/<dataset_name_raw>")
```

For example, download procesed and raw "OPFData" dataset files with:
```Python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="AI4Climate/OPFData", repo_type="dataset")
snapshot_download(repo_id="AI4Climate/OPFData_raw")
```

## Using `wget` and `curl`

Download file stored under <path_to_file> from <dataset_name>:
```bash
wget https://huggingface.co/datasets/AI4Climate/<dataset_name>/resolve/main/<path_to_file>
```

