from prefect import flow, task
import kagglehub

@task
def download_dataset(url: str, _output_dir: str):
    kagglehub.dataset_download(url, output_dir=_output_dir)

@flow
def collect_datasets(dataset_urls: list, _output_dir: str):
    for url in dataset_urls:
        download_dataset(url, _output_dir)

urls = ["willianoliveiragibin/greenhouse-gas-emissions",
        "ankushpanday1/global-warming-dataset-195-countries-1900-2023"]

collect_datasets(urls, "./data")