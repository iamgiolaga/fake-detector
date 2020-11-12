import pandas as pd

# 1. read to see if there exists a experiments.csv file
# 2. if yes, read it and store it in a dataframe
# 3. if not, raise an exception

try:
    experiments = pd.read_csv("experiments.csv")
except:
    raise FileNotFoundError

print(experiments.head(10))