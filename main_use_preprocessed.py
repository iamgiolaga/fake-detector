import pickle
import os
import pandas as pd

from classes.preprocessing import Preprocessing
from datetime import datetime

''' DESCRIPTION '''
''' This is the file where preprocessed datasets are loaded and reused '''

''' Generated data case '''

PATH_TEXTS = "generated_text/31.01.2021_19.07/"

directory = os.fsencode(PATH_TEXTS)

date = datetime.now().strftime('%d.%m.%Y')
time = datetime.now().strftime('%H.%M')

generated_data = pd.DataFrame(columns=["text", "membership"])

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        generated = pd.read_csv(os.path.join(PATH_TEXTS, filename))
        del generated["Unnamed: 0"]
        generated["text"] = generated[generated.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        generated["text"] = generated["text"].apply(lambda x: x.split(" "))

        pp_generated = Preprocessing(
            generated["text"],
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
        dataframe = pd.DataFrame(gen.aggregated)
        dataframe["membership"] = filename.split("_")[1].replace(".csv", "")
        generated_data = generated_data.append(dataframe)
        continue
    else:
        continue

preprocessing = Preprocessing(generated_data, date, time)
dataset = preprocessing.shuffle(generated_data).reset_index(drop=True)

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