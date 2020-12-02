import pandas as pd

''' DESCRIPTION '''
''' This is the file where we can visualize the final results '''

# 1. read to see if there exists a models.csv file
# 2. if yes, read it and store it in a dataframe
# 3. if not, raise an exception
from termcolor import colored

try:
    models = pd.read_csv("selected_models/models.csv")
except:
    raise FileNotFoundError

if models.empty:
    print(colored("No models found", "red"))

# TODO: prepare query that could be useful to retrieve results on some argument basis

print(models.columns)
print(models.head(10))
print(models.iloc[0])

