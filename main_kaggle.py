from kaggle.api.kaggle_api_extended import KaggleApi
from termcolor import colored

'''Authenticating With API Server'''
api = KaggleApi()
api.authenticate()

'''Downloading Datasets'''
# Download all files of a dataset
# Signature: dataset_download_files(dataset, path=None, force=False, quiet=True, unzip=False)
api.dataset_download_files('clmentbisaillon/fake-and-real-news-dataset',
                           path = "datasets/fakeandreal/",
                           unzip=True)

print(colored("Dataset correctly downloaded!", "green"))
