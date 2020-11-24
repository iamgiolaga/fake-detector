import pandas as pd
import ast
import numpy as np
from mulearn import FuzzyInductor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from classes.model import Model
from mulearn import FuzzyInductor, fuzzifier, kernel, optimization as opt

# MODEL SELECTION

''' DESCRIPTION '''
''' This is the file where models are selected by using the training set '''

def simple_split(dataset):
    X = dataset.iloc[:, 0].values  # x-component
    y = dataset.iloc[:, 1].values  # labels
    return X, y

PATH_TEXTS = "preprocessed_datasets/text/23.11.2020_02.22/final_text_dataset_38592.csv"
outer_folds = 5
inner_folds = 4

# data loading
dataset = pd.read_csv(PATH_TEXTS)

# extract sample
dataset = dataset.head(10)

X, y = simple_split(dataset)
X = [ast.literal_eval(i) for i in X] # this is needed to parse strings
X = np.array(X)

scalers = [StandardScaler(), RobustScaler(), MinMaxScaler()]

pipe = Pipeline([
    ("scaler", None),
    ("learning_algorithm", FuzzyInductor())
])

sigmas = [.225,.5]
learning_params = {
    'c': [1,10,100],
    'k': [kernel.GaussianKernel(s) for s in sigmas]
}

params = {"scaler": scalers}

for k in learning_params:
    params["learning_algorithm__" + k] = learning_params[k]

print(params)

outer_fold = StratifiedKFold(n_splits = outer_folds)
scores = []
best_models = []
best_params = []

for i, (train_id, test_id) in enumerate(outer_fold.split(X, y)):
    X_train, X_test = X[train_id], X[test_id]
    y_train, y_test = y[train_id], y[test_id]

    gs = GridSearchCV(
        pipe,
        params,
        verbose = 0,
        cv = inner_folds,
        scoring = "neg_root_mean_squared_error"
    )

    gs = gs.fit(X_train, y_train)

    score = gs.score(X_test, y_test)
    scores.append(score)
    best_models.append(gs.best_estimator_)
    best_params.append(gs.best_params_)

print(best_models)
print(best_params)

# Whenever a model is selected, an instance of Model is created and then written
f = FuzzyInductor()
m = Model(f)
m.write_model()

''' Cross Validation - tuning of c and k'''
'''
sigmas = [.225,.5]
parameters = {'c': [1,10,100],
              'k': [kernel.GaussianKernel(s) for s in sigmas]}

with warnings.catch_warnings():
    warnings.simplefilter('ignore', FitFailedWarning)

    crossval = GridSearchCV(f, param_grid=parameters)
    crossval.fit(X_train, y_train)

predictions = crossval.predict(X_test)
e.set_params(crossval.get_params())
print("Cross Validated error")
print(e.RMSE(predictions, y_test))
'''

