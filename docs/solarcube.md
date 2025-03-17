# SolarCube

## Load standardized task data

You can both download and load all data associated with standardized PowerGraph tasks
in the following fashion. For example, load the <subtask-name> sub-task
with:

```Python
from ai4climate import load_task

dataset = load_task(
    task_name='SolarCube', 
    subtask_name='<subtask-name>',
    root_path='~/AI4Climate/'
)
```

Available sub-tasks are:
- <subtask-name>


## Download raw data

You can download the raw PowerGraph datasets from 
https://zenodo.org/records/11498739. The download can be
done programmatically, and files be decompressed, using the following command

```bash
wget "https://zenodo.org/api/records/11498739/files-archive"
mv files-archive dataset.zip
unzip dataset.zip
```

