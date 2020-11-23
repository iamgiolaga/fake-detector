import pandas as pd
import os

from classes.preprocessing import Preprocessing
from datetime import datetime

# PREPROCESSING

''' DESCRIPTION '''
''' This is the file where the entire preprocessing phase is performed '''

''' FIRST SECTION: LOADING AND PREPROCESSING '''
date = datetime.now().strftime('%d.%m.%Y')
time = datetime.now().strftime('%H.%M')

## Dataset loading ## (example)
dataset_fake = pd.read_csv("datasets/fakeandreal/Fake.csv")
dataset_true = pd.read_csv("datasets/fakeandreal/True.csv")

## Detect text (news title or body) to use as input ##
analysis = "text" # let's focus on the corpus of news
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

preprocesser_fake = Preprocessing(
    fake,
    date,
    time,
    analysis = analysis,
    news_type = "fake"
) # here you can set the configuration
data_fake = preprocesser_fake.run_pipeline()
print("")
print("FINAL OUTPUT:")

if preprocesser_fake.aggregation:
    print("(TYPE: ", type(data_fake.aggregated), ")")
    print(data_fake.aggregated)

else:
    print("(TYPE: ", type(data_fake.docvectors), ")")
    print(data_fake.docvectors)

''' REAL NEWS DATASET '''

print("INPUT:")
print("(TYPE: ", type(true), ")")
print(true.head(10))

preprocesser_true = Preprocessing(
    true,
    date,
    time,
    analysis = analysis,
    news_type = "true"
)
data_true = preprocesser_true.run_pipeline()
print("")

if preprocesser_true.aggregation:
    print("(TYPE: ", type(data_true.aggregated), ")")
    print(data_true.aggregated)

else:
    print("(TYPE: ", type(data_true.docvectors), ")")
    print(data_true.docvectors)

''' SECOND SECTION: PUT TOGETHER FAKE AND REAL NEWS WITH LABEL '''

dataset_fake = pd.DataFrame(data_fake.aggregated)
dataset_true = pd.DataFrame(data_true.aggregated)

dataset_fake["label"] = 1
dataset_true["label"] = 0

dataset = preprocesser_true.prepare_dataset(dataset_fake, dataset_true)
print("DATASET IS READY")
print(dataset.head(10))

cardinality = len(dataset_true) + len(dataset_fake)

outdir = 'preprocessed_datasets/'
outname = "other"

if analysis == "text":
    outdir = "preprocessed_datasets/text/" + date + "_" + time
    outname = "final_text_dataset_" + str(cardinality) + ".csv"
else:
    if analysis == "title":
        outname = "final_title_dataset_" + str(cardinality) + ".csv"

if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)
dataset.to_csv(fullname, index=False)

preprocesser_true.write_preprocessed_dataset()
preprocesser_fake.write_preprocessed_dataset()

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
