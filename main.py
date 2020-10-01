import pandas as pd

from preprocessing import preprocessing

dataset_fake = pd.read_csv("datasets/fakeandreal/Fake.csv")
dataset_true = pd.read_csv("datasets/fakeandreal/True.csv")

titles = dataset_fake["title"]
texts = dataset_fake["text"]

preprocesser = preprocessing(texts)



