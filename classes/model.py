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

    def select_model(self, dataset, solver = "gurobi", n_iter = None, outer_folds = 5, inner_folds = 4, write = False):
        self.dataset = dataset
        self.X, self.y = self.simple_split(dataset)
        self.X = [ast.literal_eval(i) for i in self.X]  # this is needed to parse strings
        self.X = np.array(self.X)
        self.solver = solver
        self.n_iter = n_iter
        self.write = write

        ''' CONFIGURE FUZZY INDUCTOR '''
        # solver strategy
        if self.solver == "tensorflow":
            if self.n_iter is not None:
                self.strategy = (opt.solve_optimization_tensorflow, {'n_iter': self.n_iter})
            else:
                self.strategy = (opt.solve_optimization_tensorflow, {})
        else:
            if self.solver == "gurobi":
                if self.n_iter is not None:
                    self.strategy = (opt.solve_optimization_gurobi, {'n_iter': self.n_iter})
                else:
                    self.strategy = (opt.solve_optimization_gurobi, {'adjustment': 50})

        fuzzifier_type = (fuzzifier.LinearFuzzifier, {})

        # scalers = [StandardScaler(), RobustScaler(), MinMaxScaler()]

        pipe = Pipeline([
            #("scaler", None),
            ("learning_algorithm", FuzzyInductor(
                solve_strategy = self.strategy,
                fuzzifier= fuzzifier_type
            ))
        ])

        # sigmas = np.linspace(.1, 1.0, 10)
        # alphas = np.linspace(.1, 1.0, 10)
        sigmas = (0.1, 0.2, 0.3, 0.4, 0.5)
        alphas = (0.2, 0.7)
        # exponential_fuzzifiers = [
        #     (fuzzifier.ExponentialFuzzifier,
        #      {'profile': 'alpha', 'alpha': alpha}) for alpha in alphas
        # ]
        # self.fuzzifier_types = [(fuzzifier.LinearFuzzifier, {})]
        #
        # for e in exponential_fuzzifiers:
        #     self.fuzzifier_types.append(e)

        c_vector = np.logspace(0.1, 1, 10, endpoint=True)

        learning_params = {
            'c': c_vector,
            'k': [kernel.GaussianKernel(s) for s in sigmas]
            #'fuzzifier': self.fuzzifier_types
        }

        params = {}
        #params = {"scaler": scalers}

        for j in learning_params:
            params["learning_algorithm__" + j] = learning_params[j]

        outer_fold = StratifiedKFold(n_splits=outer_folds)
        best_models = []
        best_params = []
        all_scores = []

        for i, (train_id, test_id) in enumerate(outer_fold.split(self.X, self.y)):
            print("Working on fold " + str(i + 1) + " of " + str(outer_folds))
            X_train, X_test = self.X[train_id], self.X[test_id]
            y_train, y_test = self.y[train_id], self.y[test_id]

            gs = GridSearchCV(
                pipe,
                params,
                verbose=0,
                cv=inner_folds
            )

            gs = gs.fit(X_train, y_train)
            # f = gs.best_estimator_["learning_algorithm"]
            scores = self.evaluate_model(gs, X_test, y_test)

            all_scores.append(scores)
            best_models.append(gs.best_estimator_)
            best_params.append(gs.best_params_)

        self.best_models = best_models
        self.best_params = best_params
        self.all_scores = all_scores

        if self.write:
            # when the best models are found, they are serialized and stored
            self.write_model()

    def evaluate_model(self, gs, X, y):
        scores = {}
        memberships = gs.predict(X)
        predicted_labels = self.threshold(memberships)
        labels = y
        TP, FP, FN, TN = self.confusion_matrix(labels, predicted_labels, memberships)
        precision = self.precision(TP, FP)
        recall = self.recall(TP, FN)
        f1 = self.f1(precision, recall)

        scores["precisions"] = precision
        scores["recalls"] = recall
        scores["f1s"] = f1

        return scores


    def simple_split(self, dataset):
        X = dataset.iloc[:, 0].values  # x-component
        y = dataset.iloc[:, 1].values  # labels
        return X, y

    def threshold(self, membership, n = 0.5):
        memberships = []

        for x in membership:
            if x >= n:
                memberships.append(1) # fake
            else:
                memberships.append(0) # true

        return memberships

    def confusion_matrix(self, labels, predicted_labels, memberships):
        '''
        true positive: predicted_labels = labels = 1
        false positive: predicted_labels = 1 != labels = 0
        true negative: predicted_labels = labels = 0
        false negative: predicted_labels = 0 != labels = 1

        All these values are weighted by their membership
        '''

        TP = 0.0
        FP = 0.0
        FN = 0.0
        TN = 0.0

        for label, predicted_label, membership in zip(labels, predicted_labels, memberships):
            if predicted_label == 1: # positive
                weight = membership
                if predicted_label == label: # true positive
                    TP = TP + (1 * weight)
                else: # false positive
                    FP = FP + (1 * weight)
            else: # negative
                weight = 1 - membership
                if predicted_label == label: # true negative
                    TN = TN + (1 * weight)
                else: # false negative
                    FN = FN + (1 * weight)

        return TP, FP, FN, TN

    def precision(self, TP, FP):
        ''' true positive / (true positive + false positive)'''
        if TP + FP == 0.0:
            return 0.0
        else:
            return TP / (TP + FP)

    def recall(self, TP, FN):
        ''' true positive / (true positive + false negative) '''
        if TP + FN == 0.0:
            return 0.0
        else:
            return TP / (TP + FN)

    def f1(self, precision, recall):
        ''' 2 * [(precision * recall) / (precision + recall)] '''
        if precision + recall == 0.0:
            return 0.0
        else:
            return 2 * ((precision * recall) / (precision + recall))

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
            column_names = ["Date", "Time", "Sample", "Solver", "Params", "Scores"]
            models = pd.DataFrame(columns=column_names)

        for i, k in enumerate(self.best_params):
            models = models.append({
                "Date": self.date, "Time": self.time, "Sample": len(self.dataset),
                "Solver": self.solver, "Params": k, "Scores": self.all_scores[i]
            }, ignore_index=True)

        models.to_csv(fullname, index=False)

        for i, model in enumerate(self.best_models):
            outdir = "selected_models/" + self.date + "_" + self.time + "/"
            outname = "model_(" + str(i) + ")"

            if not os.path.exists(outdir):
                os.makedirs(outdir)

            fullname = os.path.join(outdir, outname)

            file = open(fullname, "wb")
            pickle.dump(model, file)
            file.close()