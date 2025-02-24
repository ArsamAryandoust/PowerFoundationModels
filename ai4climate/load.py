import os
import requests
import subprocess
import concurrent.futures

# individual task loading modules
import opfdata

def load_task(
    task_name: str,
    subtask_name: str,
    root_path: str = None,
    data_frac: int = 1,
    max_workers: int = 1024,
    max_workers_download: int = 4,
):
    """Download task repository and load standardized subtask."""
    print(f"Processing '{subtask_name}' for '{task_name}'...")

    # If no root_path is given, default to ~/AI4Climate
    if root_path is None:
        root_path = os.path.expanduser('~/AI4Climate')

    # set path to local directory
    local_dir = os.path.join(root_path, task_name)

    # download task repository
    _download_hf_repo(
        local_dir, 
        task_name,
        max_workers,
        max_workers_download
    )

    # load subtask (replace with your actual logic)
    (
        train_data, 
        val_data, 
        test_data
    ) = _load_subtask(
        local_dir, 
        subtask_name, 
        data_frac, 
        max_workers
    )
    
    return train_data, val_data, test_data


def _load_subtask(
    local_dir: str, 
    subtask_name: str,
    data_frac: int,
    max_workers: int
):
    """Load standardized task."""
    print(f"Preparing subtask '{subtask_name}'...")
    if 'OPFData' in local_dir:
        (
            train_data, 
            val_data, 
            test_data 
        ) = opfdata.load(
            local_dir, 
            subtask_name,
            data_frac,
            max_workers
        )
    else:
        raise NotImplementedError("Other tasks not yet implemented!")

    print(f"Data for {subtask_name} successfully loaded.\n")
    return train_data, val_data, test_data


def _download_hf_repo(
    local_dir: str, 
    task_name: str, 
    max_workers: int,
    max_workers_download: int
):
    """Download and uncompress all files from the Hugging Face in parallel."""
    print(f"Preparing local data directory for '{task_name}'...")

    repo_id = f'AI4Climate/{task_name}'

    # Step 1: Collect all files (recursively)
    files_to_download = []
    _collect_files(repo_id, local_dir, subpath="", files_list=files_to_download)

    # Step 2: Download them in parallel
    print(f"\nFound {len(files_to_download)} files to download.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_download) as executor:
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

    # Step 3: Uncompress files and delete compressed files in parallel
    # Detect compressed files from the downloaded list
    compressed_exts = (".zip", ".tar.gz", ".tar")
    compressed_files = [local_path for (_, local_path) in files_to_download 
                        if local_path.endswith(compressed_exts)]

    if compressed_files:
        print(f"Uncompressing {len(compressed_files)} files in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_compressed = {
                executor.submit(_uncompress_and_delete_file, path): path 
                for path in compressed_files
            }
            for future in concurrent.futures.as_completed(future_to_compressed):
                path = future_to_compressed[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Uncompress & delete failed for {path}: {e}")
        
        print("All compressed files have been uncompressed (and deleted).")


def _collect_files(
    repo_id: str, 
    local_dir: str, 
    subpath: str, 
    files_list: list
):
    """Recursively gather file paths from the Hugging Face API and append them
    as (file_url, local_entry_path) tuples to files_list.
    """
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
            file_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{entry_path}"
            files_list.append((file_url, local_entry_path))

        elif entry_type == 'directory':
            _collect_files(repo_id, local_dir, subpath=entry_path, 
                files_list=files_list)


def _download_single_file(
    url: str, 
    local_path: str
):
    """Download a single file via wget subprocess call."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"Downloading {url} -> {local_path}")
    subprocess.run(["wget", "-q", "-O", local_path, url], check=True)


def _uncompress_and_delete_file(file_path: str):
    """Decompress a file (zip/tar/tar.gz) into its directory, then delete the 
    original. Adjust or extend as needed for other compression formats.
    """
    # Choose the right tool/command based on file extension
    if file_path.endswith(".zip"):
        cmd = ["unzip", "-o", file_path, "-d", os.path.dirname(file_path)]
    elif file_path.endswith(".tar.gz"):
        cmd = ["tar", "-xzf", file_path, "-C", os.path.dirname(file_path)]
    elif file_path.endswith(".tar"):
        cmd = ["tar", "-xf", file_path, "-C", os.path.dirname(file_path)]
    else:
        print(f"Skipping unrecognized file format: {file_path}")
        return

    print(f"Uncompressing {file_path}...")
    subprocess.run(cmd, check=True)

    print(f"Deleting {file_path}...")
    os.remove(file_path)
