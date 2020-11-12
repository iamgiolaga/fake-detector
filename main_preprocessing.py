import pandas as pd
from classes.preprocessing import Preprocessing

''' DESCRIPTION '''
''' This is the file where the entire preprocessing phase is performed '''

''' FIRST SECTION: LOADING AND PREPROCESSING '''

## Dataset loading ## (example)
dataset_fake = pd.read_csv("datasets/fakeandreal/Fake.csv")
dataset_true = pd.read_csv("datasets/fakeandreal/True.csv")

## Detect text (news title or body) to use as input ##
analysis = "text"
fake = dataset_fake[analysis]
true = dataset_true[analysis]

path = "preprocessed_datasets/final_other_dataset.csv"

if analysis == "text":
    path = "preprocessed_datasets/final_text_dataset.csv"
else:
    if analysis == "title":
        path = "preprocessed_datasets/final_title_dataset.csv"

print("")
print("PREPROCESSING:")
print("")

print("INPUT:")
print("(TYPE: ", type(fake), ")")

''' FAKE NEWS DATASET '''

preprocesser_fake = Preprocessing(fake.head(1000)) # here you can set the configuration
data_fake = preprocesser_fake.run_pipeline()
print("")
print("FINAL OUTPUT:")
print("(TYPE: ", type(data_fake.aggregated), ")")
print(data_fake.aggregated)

''' REAL NEWS DATASET '''

print("INPUT:")
print("(TYPE: ", type(true), ")")
preprocesser_true = Preprocessing(true.head(1000))
data_true = preprocesser_true.run_pipeline()
print("")
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
dataset.to_csv(path, index=False)

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