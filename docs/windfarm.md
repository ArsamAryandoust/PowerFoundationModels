# WindFarm

## Load standardized task data

You can both download and load all data associated with standardized WindFarm
tasks in the following fashion. For example, load the `XXXX` sub-task with:

```Python
from ai4climate import load_task

dataset = load_task(
    task_name='WindFarm', 
    subtask_name='XXXX',
    root_path='~/AI4Climate/'
)
```

Available sub-tasks are:
- `XXXX`
- `XXXX`
- `XXXX`

## Download raw task data

You can download the raw WindFarm datasets from FigShare at 
https://doi.org/10.6084/m9.figshare.24798654.

Programmatically, we download the dataset with:

```bash
wget https://figshare.com/ndownloader/articles/24798654/versions/2
mv 2 WindFarm.zip
unzip WindFarm
rm WindFarm.zip
mv SDWPF_dataset WindFarm
```

