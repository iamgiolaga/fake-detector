import pandas as pd
from classes.preprocessing import Preprocessing
import os

# PREPROCESSING

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

print("")
print("PREPROCESSING:")
print("")

''' FAKE NEWS DATASET '''

print("INPUT:")
print("(TYPE: ", type(fake), ")")
print(fake.head(10))

preprocesser_fake = Preprocessing(fake.head(1000)) # here you can set the configuration
data_fake = preprocesser_fake.run_pipeline()
print("")
print("FINAL OUTPUT:")

if preprocesser_fake.aggregation:
    print("(TYPE: ", type(data_fake.aggregated_test), ")")
    print(data_fake.aggregated_test)

else:
    print("(TYPE: ", type(data_fake.docvectors_test), ")")
    print(data_fake.docvectors_test)

''' REAL NEWS DATASET '''

print("INPUT:")
print("(TYPE: ", type(true), ")")
print(true.head(10))

preprocesser_true = Preprocessing(true.head(1000))
data_true = preprocesser_true.run_pipeline()
print("")

if preprocesser_true.aggregation:
    print("(TYPE: ", type(data_true.aggregated_test), ")")
    print(data_true.aggregated_test)

else:
    print("(TYPE: ", type(data_true.docvectors_test), ")")
    print(data_true.docvectors_test)

''' SECOND SECTION: PUT TOGETHER FAKE AND REAL NEWS WITH LABEL '''

dataset_fake_train = pd.DataFrame(data_fake.aggregated_train)
dataset_true_train = pd.DataFrame(data_true.aggregated_train)
dataset_fake_test = pd.DataFrame(data_fake.aggregated_test)
dataset_true_test = pd.DataFrame(data_true.aggregated_test)

dataset_fake_train["label"] = 1
dataset_true_train["label"] = 0
dataset_fake_test["label"] = 1
dataset_true_test["label"] = 0

dataset_train = preprocesser_true.prepare_dataset(dataset_fake_train, dataset_true_train)
dataset_test = preprocesser_true.prepare_dataset(dataset_fake_test, dataset_true_test)
print("DATASET IS READY")
print(dataset_test.head(10))

cardinality_train = len(dataset_true_train) + len(dataset_fake_train)
cardinality_test =  len(dataset_true_test) + len(dataset_fake_test)

outdir = 'preprocessed_datasets/train/'
outname = "other"

if analysis == "text":
    outname = "final_text_dataset_" + str(cardinality_train) + ".csv"
else:
    if analysis == "title":
        outname = "final_title_dataset_" + str(cardinality_train) + ".csv"

if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)
dataset_train.to_csv(fullname, index=False)

outdir = 'preprocessed_datasets/test/'
outname = "other"

if analysis == "text":
    outname = "final_text_dataset_" + str(cardinality_test) + ".csv"
else:
    if analysis == "title":
        outname = "final_title_dataset_" + str(cardinality_test) + ".csv"

if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)
dataset_test.to_csv(fullname, index=False)

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