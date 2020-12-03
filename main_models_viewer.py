import pandas as pd
import ast

''' DESCRIPTION '''
''' This is the file where we can visualize the final results '''

from termcolor import colored

def select_scores(models):
    for i, score in enumerate(models["Scores"]):
        print(i, score)

def select_best_precisions(models):
    precisions = {}
    scores = models["Scores"]

    for i, score in enumerate(scores):
        precisions[i] = score['precision']

    precisions = dict(sorted(precisions.items(), key = lambda item: item[1], reverse = True))
    print(precisions)

def select_best_recalls(models):
    recalls = {}
    scores = models["Scores"]

    for i, score in enumerate(scores):
        recalls[i] = score['recall']

    recalls = dict(sorted(recalls.items(), key = lambda item: item[1], reverse = True))
    print(recalls)

def select_best_f1s(models):
    f1s = {}
    scores = models["Scores"]

    for i, score in enumerate(scores):
        f1s[i] = score['f1']

    f1s = dict(sorted(f1s.items(), key = lambda item: item[1], reverse = True))
    print(f1s)

try:
    models = pd.read_csv("selected_models/models.csv")
except:
    raise FileNotFoundError

if models.empty:
    print(colored("No models found", "red"))

models["Scores"] = [ast.literal_eval(i) for i in models["Scores"]]

select_scores(models)
select_best_precisions(models)
select_best_recalls(models)
select_best_f1s(models)

# When you want to analyze a specific experiment, you can call this
print(models.iloc[6])