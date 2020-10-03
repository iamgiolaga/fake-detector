import pandas as pd
from preprocessing import preprocessing
from doc2vec import doc2vec

## Dataset loading ## (example)
dataset_fake = pd.read_csv("datasets/fakeandreal/Fake.csv")
dataset_true = pd.read_csv("datasets/fakeandreal/True.csv")

## Detect text (news title or body) ##
titles = dataset_fake["title"]
texts = dataset_fake["text"]

## Preprocessing ##
preprocesser = preprocessing(titles.head(10), entity_recognition=True)
preprocessed = preprocesser.run_pipeline()
# print(preprocessed.entities) # extract entities from text
print(preprocessed.news)

document2vector = doc2vec(preprocessed.news)
vector = document2vector.run_doc2vec()
print(vector)