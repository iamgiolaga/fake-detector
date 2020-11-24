import pandas as pd

from classes.model import Model

# MODEL SELECTION

''' DESCRIPTION '''
''' This is the file where models are selected by doing a nested cross validation '''

PATH_TEXTS = "preprocessed_datasets/text/23.11.2020_02.22/final_text_dataset_38592.csv"

# data loading
dataset = pd.read_csv(PATH_TEXTS)

# extract sample
dataset = dataset.head(10)

# select best models
m = Model()
m.select_model(dataset)

# when the best models are found, they are serialized and stored
m.write_model()

