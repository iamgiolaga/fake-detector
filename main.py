import pandas as pd

from ppsteps import DuplicateRowsRemoval
from preprocessing import Preprocessing

## DESCRIPTION ##
# This is the file where experiments are launched

## Dataset loading ## (example)
dataset_fake = pd.read_csv("datasets/fakeandreal/Fake.csv")
dataset_true = pd.read_csv("datasets/fakeandreal/True.csv")

## Detect text (news title or body) ##
titles = dataset_fake["title"]
texts = dataset_fake["text"]

print("INPUT:")
print(titles.head(10))

## Preprocessing ##
preprocesser = Preprocessing(titles.head(10), entity_recognition = True) # here you can set the configuration
data = preprocesser.run_pipeline()
# print(preprocessed.configuration) # get configuration
# print(preprocessed.entities) # extract entities from text
processed_result = pd.DataFrame(data.preprocessed)
result = pd.DataFrame(data.wordvectors)
aggregated_result = pd.DataFrame(data.aggregated)
print("")
processed_result.to_csv("results/preprocessed.csv")
result.to_csv("results/result.csv")
aggregated_result.to_csv("results/aggregated.csv")

# for each document of the corpus
# Word2Vec takes in input a m-length vector of words and outputs m vectors of fixed k length
# where m is the number of words in the document and k is the number of features
# a further step here is to aggregate the m vectors into one of length m
print("FINAL OUTPUT:")
print(data.aggregated)

# for each document of the corpus
# Doc2Vec takes in input a m-length vector of words and outputs (x,y)
# where x is the tagged document and y is the similarity of the document with respect to the others
# so each document is represented by a 2-length vector
# ----------------------------------
# preprocesser = Preprocessing(titles.head(10), entity_recognition = True, doc2vec = True, word2vec=False) # here you can set the configuration
# data = preprocesser.run_pipeline()
# print(data.docvectors)