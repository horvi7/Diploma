#How to start the flow:
#prefect profile create local   -> következő beállításokat elmenti
#export prefect api. Alapbol cloudban akart futni de mi localban akartuk futtatni, igy ezt kellett beallitani
#prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
#ez jön vissza: updated profile local
#prefect server start (ez most már a lokal lesz)


from prefect import flow, task
import kagglehub
import os
import random

os.environ["KAGGLE_USERNAME"] = "horvthdmmrton"
os.environ["KAGGLE_KEY"] = "KGAT_2e7fe800a42750e3587f81735f08628b"
from kaggle.api.kaggle_api_extended import KaggleApi


@task
def download_dataset(url: str, output_dir: str):
    kagglehub.dataset_download(url, output_dir=output_dir)

@flow
def collect_datasets(dataset_urls: list, output_dir: str):
    for url in dataset_urls:
        download_dataset(url, output_dir)

def get_remote_version(datset_url: str):    
    api = KaggleApi()
    api.authenticate()
    
    dataset = api.dataset_list(search = "tanejanikhil/football")
    print(dataset[-1].current_version_number)
    
def get_local_version(dir: str):
    max_version = -1
    for file in os.listdir(dir):
        version = file.split("_")[-1]
        version = version.split(".")[0]
        version = int(version.replace("v", ""))

        max_version = max(max_version, version)
    return max_version

params = {
    "dataset_urls": ["tanejanikhil/football"],
    "output_dir": f"./data/{random.randint(1, 1000)}"
}

#get_remote_version("tanejanikhil/football")
print(get_local_version("./data/co2"))
#collect_datasets.serve(name="collect-data", cron="2 * * * *", parameters=params)