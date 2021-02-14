import pickle
import os
import pandas as pd
import ast

from classes.preprocessing import Preprocessing
from datetime import datetime

''' DESCRIPTION '''
''' This is the file where preprocessed datasets are loaded and reused '''

''' Generated data case '''

PATH_TEXTS = "generated_text/14.02.2021_16.46/generated_dataset_500.csv"

# directory = os.fsencode(PATH_TEXTS)

date = datetime.now().strftime('%d.%m.%Y')
time = datetime.now().strftime('%H.%M')

generated = pd.read_csv(PATH_TEXTS)
print(generated)

generated["text"] = generated["text"].apply(lambda s: ast.literal_eval(s))

pp_generated = Preprocessing(
    generated,
    date,
    time,
    analysis="text",
    news_type="generated",
    duplicate_rows_removal=False, lowercasing=False, tokenization=False,
    lemmatization=False, noise_removal=False, stemming=False,
    stopword_removal=False, entity_recognition=False, data_augmentation=False,
    word2vec=True, doc2vec=False, aggregation=True
)  # here you can set the configuration

gen = pp_generated.run_pipeline()
dataframe = pd.DataFrame(gen.aggregated, columns = ["text"])
dataframe["membership"] = generated["membership"]
dataset = pp_generated.shuffle(dataframe).reset_index()
dataset.columns = ["old index","text", "membership"]
dataset.index.name = "index"

cardinality = len(dataset)

outdir = "generated_datasets/" + date + "_" + time
outname = "generated_dataset_" + str(cardinality) + ".csv"

if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)
dataset.to_csv(fullname, index=True)

''' Classical preprocessed data case '''

# fullname = "preprocessed_datasets/text/28.12.2020_18.48/dataset_fake.pickle"
# file = open(fullname, "rb")
# dataset_fake = pickle.load(file)
#
# fullname = "preprocessed_datasets/text/28.12.2020_18.48/dataset_true.pickle"
# file = open(fullname, "rb")
# dataset_true = pickle.load(file)
#
# print(len(dataset_fake.aggregated))
# print(len(dataset_true.aggregated))