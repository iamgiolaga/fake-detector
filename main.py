import pandas as pd

from ppsteps import DuplicateRowsRemoval
from preprocessing import Preprocessing

''' DESCRIPTION '''
''' This is the file where experiments are launched '''

''' FIRST SECTION: LOADING AND PREPROCESSING '''

## Dataset loading ## (example)
dataset_fake = pd.read_csv("datasets/fakeandreal/Fake.csv")
dataset_true = pd.read_csv("datasets/fakeandreal/True.csv")

## Detect text (news title or body) to use as input ##
titles_fake = dataset_fake["title"]
titles_true = dataset_true["title"]

print("")
print("PREPROCESSING:")
print("")

print("INPUT:")
print("(TYPE: ", type(titles_fake.head(200)), ")")

''' FAKE NEWS DATASET '''

preprocesser_fake = Preprocessing(titles_fake.head(200)) # here you can set the configuration
data_fake = preprocesser_fake.run_pipeline()
processed_result = pd.DataFrame(data_fake.preprocessed)
result = pd.DataFrame(data_fake.wordvectors)
aggregated_result = pd.DataFrame(data_fake.aggregated)
print("")
processed_result.to_csv("results/preprocessed_fake.csv")
result.to_csv("results/result_fake.csv")
aggregated_result.to_csv("results/aggregated_fake.csv")
print("FINAL OUTPUT:")
print("(TYPE: ", type(data_fake.aggregated), ")")
print(data_fake.aggregated)

''' REAL NEWS DATASET '''

print("INPUT:")
print("(TYPE: ", type(titles_true.head(200)), ")")
preprocesser_true = Preprocessing(titles_true.head(200))
data_true = preprocesser_true.run_pipeline()
processed_result = pd.DataFrame(data_true.preprocessed)
result = pd.DataFrame(data_true.wordvectors)
aggregated_result = pd.DataFrame(data_true.aggregated)
print("")
processed_result.to_csv("results/preprocessed_true.csv")
result.to_csv("results/result_true.csv")
aggregated_result.to_csv("results/aggregated_true.csv")
print("FINAL OUTPUT:")
print("(TYPE: ", type(data_true.aggregated), ")")
print(data_true.aggregated)

''' SECOND SECTION: PUT TOGETHER FAKE AND REAL NEWS WITH LABEL '''

dataset_fake = pd.DataFrame(data_fake.aggregated)
dataset_true = pd.DataFrame(data_true.aggregated)
dataset_fake["label"] = 1
dataset_true["label"] = 0
dataset = preprocesser_true.prepare_dataset(dataset_fake, dataset_true)

print("DATASET IS READY")
print(dataset.head(10))
dataset.to_csv("results/final_dataset.csv", index=False)

# for each document of the corpus
# Word2Vec takes in input a m-length vector of words and outputs m vectors of fixed k length
# where m is the number of words in the document and k is the number of features
# a further step here is to aggregate the m vectors into one of length m

# for each document of the corpus
# Doc2Vec takes in input a m-length vector of words and outputs (x,y)
# where x is the tagged document and y is the similarity of the document with respect to the others
# so each document is represented by a 2-length vector
# ----------------------------------
# preprocesser = Preprocessing(titles.head(10), entity_recognition = True, doc2vec = True, word2vec=False) # here you can set the configuration
# data = preprocesser.run_pipeline()
# print(data.docvectors)

''' THIRD SECTION: LEARNING A MODEL TO RECOGNIZE FAKE NEWS '''
print("")
print("LEARNING")