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

@task
def download_dataset(url: str, output_dir: str):
    kagglehub.dataset_download(url, output_dir=output_dir)

@flow
def collect_datasets(dataset_urls: list, output_dir: str):
    for url in dataset_urls:
        download_dataset(url, output_dir)


params = {
    "dataset_urls": ["tanejanikhil/football"],
    "output_dir": f"./data/{random.randint(1, 1000)}"
}

collect_datasets.serve(name="collect-data", cron="2 * * * *", parameters=params)