#Prefect el eltt hagyva
import json
import os

import kagglehub
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

os.environ["KAGGLE_USERNAME"] = "horvthdmmrton"
os.environ["KAGGLE_KEY"] = "KGAT_2e7fe800a42750e3587f81735f08628b"

def collect_datasets():
    sources = load_sources()
    for name, url in sources.items():
        print(f"Checking {name} dataset...")
        should_download, version = should_download_dataset(name, url)
       
        if should_download:
            print(f"Downloading {name} dataset...")
            download_dataset(url, name, version)
        else:
            print(f"{name} dataset is up to date.")

def should_download_dataset(name: str, url: str) -> bool:
    local_version = get_local_version(name)
    remote_version = get_remote_version(url)

    if remote_version is None:
        return True, 1
    
    if local_version < remote_version:
        return True, remote_version

    return False, None

def download_dataset(url: str, output_dir: str, version: int):
    output_dir = f"./data/{output_dir}/{output_dir}_v{version}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    kagglehub.dataset_download(url, output_dir=output_dir,)

def get_remote_version(dataset_url: str):    
    api = KaggleApi()
    api.authenticate()
    
    dataset = api.dataset_list(search = dataset_url)

    if(len(dataset) == 0):
        print(f"No dataset found for {dataset_url}.")
        return None

    return dataset[-1].current_version_number

def get_local_version(name: str) -> int:
    base_path = f"./data/{name}"

    if not os.path.exists(base_path):
        print(f"No local versions found for {name}.")
        return -1

    print(f"Checking local versions in {base_path}...")

    max_version = -1

    for folder in os.listdir(base_path):
        if folder.startswith(f"{name}_v"):
            version = int(folder.split("_v")[-1])
            max_version = max(max_version, version)

    return max_version

def get_latest_as_df(name_str: str):
    base_path = f"./data/{name_str}"
    if not os.path.exists(base_path):
        print(f"No local versions found for {name_str}.")
        return None

    max_version = get_local_version(name_str)
    if max_version == -1:
        print(f"No local versions found for {name_str}.")
        return None

    latest_version_path = f"{base_path}/{name_str}_v{max_version}"
    
    # Assuming the dataset is in CSV format, adjust if necessary
    csv_files = [f for f in os.listdir(latest_version_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {latest_version_path}.")
        return None
    
    latest_csv_path = os.path.join(latest_version_path, csv_files[0])
    
    return pd.read_csv(latest_csv_path)

def load_sources(path: str = "./data/sources.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)

collect_datasets()