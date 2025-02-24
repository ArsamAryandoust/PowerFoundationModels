"""Load requested task from AI4Climate Hugging Face Hub repository."""
import os
import requests
import subprocess
import concurrent.futures

def load_task(
    task_name: str,
    subtask_name: str,
    root_path: str = None,
    max_workers: int = 4
):
    """
    Load a subtask's data from a Hugging Face dataset (AI4Climate/<task_name>)
    and download all files in parallel using up to `max_workers` threads.
    """
    print(f"Loading '{subtask_name}' from '{task_name}'...")

    # If no root_path is given, default to ~/AI4Climate
    if root_path is None:
        root_path = os.path.expanduser('~/AI4Climate')

    # Full local path for the dataset
    local_dir = os.path.join(root_path, task_name)
    os.makedirs(local_dir, exist_ok=True)  # Ensure base directory exists

    # Construct the Hugging Face repository ID
    repo_id = f'AI4Climate/{task_name}'

    # Step 1: Collect all files (recursively)
    files_to_download = []
    _collect_files(repo_id, local_dir, subpath="", files_list=files_to_download)

    # Step 2: Download them in parallel
    print(f"\nFound {len(files_to_download)} files to download.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(_download_single_file, url, local_path): (url, local_path)
            for (url, local_path) in files_to_download
        }
        for future in concurrent.futures.as_completed(future_to_file):
            url, local_path = future_to_file[future]
            try:
                future.result()  # will raise CalledProcessError if 'wget' fails
            except Exception as e:
                print(f"Download failed for {url}: {e}")

    print(f"Data for {repo_id} successfully downloaded to {local_dir}.\n")



def _collect_files(repo_id: str, local_dir: str, subpath: str, files_list: list):
    """
    Recursively gather file paths from the Hugging Face API and append them
    as (file_url, local_entry_path) tuples to files_list.
    """
    # Construct the API endpoint
    api_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main"
    if subpath:
        api_url += f"/{subpath}"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching file list: {e}")
        return

    content_list = response.json()
    
    for entry in content_list:
        entry_type = entry['type']
        entry_path = entry['path']
        local_entry_path = os.path.join(local_dir, entry_path)

        if entry_type == 'file':
            # Build file URL
            file_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{entry_path}"
            files_list.append((file_url, local_entry_path))

        elif entry_type == 'directory':
            # Recursive call to process subdirectories
            _collect_files(repo_id, local_dir, subpath=entry_path, files_list=files_list)


def _download_single_file(url: str, local_path: str):
    """Download a single file via wget subprocess call."""
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    print(f"Downloading {url} -> {local_path}")
    
    subprocess.run(["wget", "-q", "-O", local_path, url], check=True)