"""Load requested task from AI4Climate Hugging Face Hub repository."""
import os
import sys

def load_task(
    task_name: str,
    subtask_name: str,
    root_path: str = None
):
    """ """
    print(f"Loading '{subtask_name}' from '{task_name}'...")

    # If no root_path is given, default to ~/AI4Climate
    if root_path is None:
        root_path = os.path.expanduser('~/AI4Climate')

    # Full local path for the dataset
    task_path = os.path.join(root_path, task_name)

    # Download the repository to the local path
    _download_data(task_name, task_path)


def _download_data(
    task_name: str,
    task_path: str
):
    """Download task repository containing required data and testbeds."""
    # Construct the Hugging Face repository ID
    repo_id = f'AI4Climate/{task_name}'

    # Create local directory if it doesn't exist
    os.makedirs(task_path, exist_ok=True)

    # Download entire repository snapshot from Hugging Face
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=task_path,
    )

    print(f"Task '{task_name}' data successfully downloaded to '{task_path}'.")


