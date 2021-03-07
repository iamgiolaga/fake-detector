import pandas as pd

from classes.model import Model

# MODEL SELECTION

''' DESCRIPTION '''
''' This is the file where models are selected by doing a nested cross validation '''

PATH_TEXTS = "generated_datasets/14.02.2021_17.44/generated_dataset_500.csv"
#PATH_TEXTS = "preprocessed_datasets/text/28.12.2020_18.48/final_text_dataset_38592.csv"
#PATH_TEXTS = "preprocessed_datasets/text/06.02.2021_21.00/final_text_dataset_7926.csv"

# data loading
dataset = pd.read_csv(PATH_TEXTS)

# code for generated datasets
del dataset["index"]

# extract sample
#dataset = dataset.head(10)

print("INPUT")
print(dataset.head(10))

# select the best models
m = Model()
m.select_model(dataset.head(100), write=False)

