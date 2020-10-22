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
preprocesser = Preprocessing(titles.head(10), entity_recognition = True, doc2vec = True, word2vec=False) # here you can set the configuration
data = preprocesser.run_pipeline()
# print(preprocessed.configuration) # get configuration
# print(preprocessed.entities) # extract entities from text
print(data.preprocessed)
print("")

# Doc2Vec takes in input a n x m matrix and outputs a n-length vector of tuples (x,y)
# where x is the tagged document and y is the similarity of the document with respect to the others
# ----------------------------------
# preprocesser = Preprocessing(titles.head(10), entity_recognition = True, doc2vec = True, word2vec=False) # here you can set the configuration
# data = preprocesser.run_pipeline()
# print(data.docvectors)