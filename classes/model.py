import pandas as pd
import os
import pickle
import ast
import numpy as np
from mulearn import FuzzyInductor, fuzzifier, kernel, optimization as opt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from datetime import datetime

class Model:

    def __init__(self):
        self.date = datetime.now().strftime('%d.%m.%Y')
        self.time = datetime.now().strftime('%H.%M')

    def select_model(self, dataset, outer_folds = 5, inner_folds = 4):
        X, y = self.simple_split(dataset)
        X = [ast.literal_eval(i) for i in X]  # this is needed to parse strings
        X = np.array(X)

        scalers = [StandardScaler(), RobustScaler(), MinMaxScaler()]

        pipe = Pipeline([
            ("scaler", None),
            ("learning_algorithm", FuzzyInductor())
        ])

        sigmas = [.225, .5]
        learning_params = {
            'c': [1, 10, 100],
            'k': [kernel.GaussianKernel(s) for s in sigmas]
        }

        params = {"scaler": scalers}

        for k in learning_params:
            params["learning_algorithm__" + k] = learning_params[k]

        outer_fold = StratifiedKFold(n_splits=outer_folds)
        scores = []
        best_models = []
        best_params = []

        for i, (train_id, test_id) in enumerate(outer_fold.split(X, y)):
            X_train, X_test = X[train_id], X[test_id]
            y_train, y_test = y[train_id], y[test_id]

            gs = GridSearchCV(
                pipe,
                params,
                verbose=0,
                cv=inner_folds,
                scoring="neg_root_mean_squared_error"
            )

            gs = gs.fit(X_train, y_train)

            score = gs.score(X_test, y_test)
            scores.append(score)
            best_models.append(gs.best_estimator_)
            best_params.append(gs.best_params_)

        self.best_models = best_models
        self.best_params = best_params


    def simple_split(self, dataset):
        X = dataset.iloc[:, 0].values  # x-component
        y = dataset.iloc[:, 1].values  # labels
        return X, y

    def write_model(self):
        # 1. read to see if any dataframe is already available
        # 2. if yes, read it and update it
        # 3. if not, create it and fill it

        outdir = "selected_models/"
        outname = "models.csv"

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        fullname = os.path.join(outdir, outname)

        try:
            models = pd.read_csv("selected_models/models.csv")
        except:
            column_names = ["Date", "Time", "Params"]
            models = pd.DataFrame(columns=column_names)

        for k in self.best_params:
            models = models.append({
                "Date": self.date, "Time": self.time, "Params": k
            }, ignore_index=True)

            models.to_csv(fullname, index=False)

        for i, model in enumerate(self.best_models):
            outdir = "selected_models/"
            outname = "model_" + self.date + "_" + self.time + "(" + str(i) + ")"

            if not os.path.exists(outdir):
                os.makedirs(outdir)

            fullname = os.path.join(outdir, outname)

            file = open(fullname, "wb")
            pickle.dump(model, file)
            file.close()