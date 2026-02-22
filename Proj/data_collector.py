from prefect import flow, task
import kagglehub

@task
def download_dataset(url: str, output_dir: str):
    kagglehub.dataset_download(url, output_dir=output_dir)

@flow
def collect_datasets(dataset_urls: list, output_dir: str):
    for url in dataset_urls:
        download_dataset(url, output_dir)

