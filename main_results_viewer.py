import pandas as pd

# 1. read to see if there exists a experiments.csv file
# 2. if yes, read it and store it in a dataframe
# 3. if not, raise an exception
from termcolor import colored

try:
    experiments = pd.read_csv("results/experiments.csv")
except:
    raise FileNotFoundError

if experiments.empty:
    print(colored("No experiments found", "red"))

print(experiments.columns)
print(experiments.head(10))
print(experiments.iloc[0])

