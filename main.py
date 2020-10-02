import pandas as pd
import spacy

from preprocessing import preprocessing

## Dataset loading ## (example)
dataset_fake = pd.read_csv("datasets/fakeandreal/Fake.csv")
dataset_true = pd.read_csv("datasets/fakeandreal/True.csv")

## Detect text (news title or body) ##
titles = dataset_fake["title"]
texts = dataset_fake["text"]

## Preprocessing ##
preprocesser = preprocessing(titles)
preprocesser.run_pipeline()

news = preprocesser.get_news()
print(news)