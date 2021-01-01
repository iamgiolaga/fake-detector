import pandas as pd

from classes.model import Model

# MODEL SELECTION

''' DESCRIPTION '''
''' This is the file where models are selected by doing a nested cross validation '''

PATH_TEXTS = "preprocessed_datasets/text/28.12.2020_18.48/final_text_dataset_38592.csv"

# data loading
dataset = pd.read_csv(PATH_TEXTS)

# extract sample
dataset = dataset.head(500)

print("INPUT")
print(dataset.head(10))

# select the best models
m = Model()
m.select_model(dataset, write=True)

