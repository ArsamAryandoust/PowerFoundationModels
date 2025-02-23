"""Load requested task from AI4Climate Hugging Face Hub repository.

This is the main API to the library.

Example use:

from ai4climate import load
train, valid, test = load.load_task(
    'OPFData',
    'train_small_test_medium'
)

"""
import os

def load_task(
    task_name: str,
    subtask_name: str,
    root_path: str = None
):
    """ """
    print(f"Loading {subtask_name} from {task_name}")

    # set home directory for root named 'AI4Climate'
    if root_path is None:
        root_path = '~/AI4Climate'

    # set path to dataset
    task_path = os.path.join(root_path, task_name)

    # download dataset
    _download_data(task_name, task_path)


def _download_data(
    task_name: str,
    task_path: str
):
    """Download task repository containing required data and testbeds."""

    # set reposiory ID
    base_repo = 'AI4Climate'
    repo_id = os.path.join(base_repo, task_name)

    # create directory if not existent already.
    if not os.isdir(task_path):
        os.mkdir(task_path)

    # download entire repository
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=task_path)

