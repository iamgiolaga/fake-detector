import pandas as pd
import os
from tqdm.auto import tqdm

''' DESCRIPTION '''
''' This is the file where raw datasets are prepared in order to have the same format '''

''' Dataset Profner '''

PATH_TEXTS = "raw_datasets/profner/txt-files/train/"
directory = os.fsencode(PATH_TEXTS)

raw_data_train = pd.DataFrame(columns=["text", "id"])

for file in tqdm(os.listdir(directory)):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"):
        f = open(os.path.join(PATH_TEXTS, filename), "r")
        id_data = filename.split(".")[0]
        data = f.read()
        raw_data_train = raw_data_train.append({"text": data, "id": id_data}, ignore_index = True)
        print(" ", end = "\r")

ground_truth_train = pd.read_csv("raw_datasets/profner/subtask-1/train.tsv", sep = "\t").sort_values("tweet_id")
raw_data_train = raw_data_train.join(ground_truth_train["label"])

# split in fake and true
fake_train = raw_data_train.loc[raw_data_train['label'] == 1]
true_train = raw_data_train.loc[raw_data_train['label'] == 0]

PATH_TEXTS = "raw_datasets/profner/txt-files/valid/"
directory = os.fsencode(PATH_TEXTS)

raw_data_valid = pd.DataFrame(columns=["text", "id"])
print("")

for file in tqdm(os.listdir(directory)):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"):
        f = open(os.path.join(PATH_TEXTS, filename), "r")
        id_data = filename.split(".")[0]
        data = f.read()
        raw_data_valid = raw_data_valid.append({"text": data, "id": id_data}, ignore_index = True)
        print(" ", end = "\r")

ground_truth_valid = pd.read_csv("raw_datasets/profner/subtask-1/valid.tsv", sep = "\t").sort_values("tweet_id")
raw_data_valid = raw_data_valid.join(ground_truth_valid["label"])

# split in fake and true
fake_valid = raw_data_valid.loc[raw_data_valid['label'] == 1]
true_valid = raw_data_valid.loc[raw_data_valid['label'] == 0]

fake = pd.concat([fake_train, fake_valid])
true = pd.concat([true_train, true_valid])

outdir = "datasets/profner/"
outname = "Fake.csv"

if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)
fake.to_csv(fullname, index=True)

outname = "True.csv"

if not os.path.exists(outdir):
    os.makedirs(outdir)

fullname = os.path.join(outdir, outname)
true.to_csv(fullname, index=True)
