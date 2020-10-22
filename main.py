import pandas as pd
from preprocessing import Preprocessing

## DESCRIPTION ##
# This is the file where experiments are launched

## Dataset loading ## (example)
dataset_fake = pd.read_csv("datasets/fakeandreal/Fake.csv")
dataset_true = pd.read_csv("datasets/fakeandreal/True.csv")

## Detect text (news title or body) ##
titles = dataset_fake["title"]
texts = dataset_fake["text"]

## Preprocessing ##
preprocesser = Preprocessing(titles.head(10), entity_recognition = True) # here you can set the configuration
data = preprocesser.run_pipeline()
# print(preprocessed.configuration) # get configuration
# print(preprocessed.entities) # extract entities from text
print(data.preprocessed)
print("")
print(data.wordvectors)