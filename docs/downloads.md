# Downloads

We host both the raw and processed datasets of the AI4Climate collection on
Hugging Face Hub. Users can download both of these using either Hugging Face's
python package called `datasets` or by directly accessing the dataset files via 
a URL.

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

## Using Direct Download via URL

Download <sample_data_file.zip> from <dataset_name> as <dataset_name_filename>
using `wget`:
```bash
wget https://huggingface.co/AI4Climate/<dataset_name>/tree/main/<sample_data_file.zip> -O <dataset_name_filename>.zip
```

or `curl`:
```bash
curl -L https://huggingface.co/AI4Climate/<dataset_name>/tree/main/<sample_data_file.zip> -O <dataset_name_filename>.zip
```
