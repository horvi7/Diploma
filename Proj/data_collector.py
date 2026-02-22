from prefect import flow, task
import kagglehub

@task
def download_dataset(url: str, path: str):
    kagglehub.download(url, path)

@flow
def collect_datasets(dataset_urls: list, path: str):
    for url in dataset_urls:
        download_dataset(url, path)

urls = ["willianoliveiragibin/greenhouse-gas-emissions",
        "ankushpanday1/global-warming-dataset-195-countries-1900-2023"]

collect_datasets(urls, "./data")