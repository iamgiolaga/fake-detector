import pandas as pd
import os
import ast
import numpy as np
import dill

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
        self.id, self.X, self.y = self.simple_split(dataset)
        self.X = [ast.literal_eval(i) for i in self.X]  # this is needed to parse strings
        self.X = np.array(self.X)
        self.solver = solver
        self.n_iter = n_iter
        self.write = write

        ''' SET LOOKUP TABLE FOR FUTURE RETRIEVE OF NEWS '''
        ''' 
        lookup = {
            <feature_vector> = {
                'id' = <id_row>
                'label' = <label>
            }
        }
        '''
        self.lookup = {}

        for id, x, label in zip(self.id, self.X, self.y):
            inner_lookup = {}
            inner_lookup["id"] = id
            inner_lookup["label"] = label
            self.lookup[str(x)] = inner_lookup

        ''' CONFIGURE FUZZY INDUCTOR '''
        # solver strategy
        if self.solver == "tensorflow":
            if self.n_iter is not None:
                self.strategy = opt.TensorFlowSolver(n_iter=self.n_iter)
            else:
                self.strategy = opt.TensorFlowSolver()
        else:
            if self.solver == "gurobi":
                self.strategy = opt.GurobiSolver()

        fuzzifier_type = fuzzifier.LinearFuzzifier()

        # scalers = [StandardScaler(), RobustScaler(), MinMaxScaler()]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("learning_algorithm", FuzzyInductor(
                solver = self.strategy,
                fuzzifier = fuzzifier_type
            ))
        ])

        # sigmas = np.linspace(.1, 1.0, 10)
        # alphas = np.linspace(.1, 1.0, 10)
        # sigmas = (0.1, 0.2, 0.3, 0.4, 0.5)
        sigmas = np.logspace(-5, 5, 6, endpoint=True)
        alphas = (0.2, 0.7)
        # exponential_fuzzifiers = [
        #     (fuzzifier.ExponentialFuzzifier,
        #      {'profile': 'alpha', 'alpha': alpha}) for alpha in alphas
        # ]
        # self.fuzzifier_types = [(fuzzifier.LinearFuzzifier, {})]
        #
        # for e in exponential_fuzzifiers:
        #     self.fuzzifier_types.append(e)

        c_vector = [0.01, 0.1, 0.5, 1.0, 10, 100]

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

        folds = {}
        best_models = []
        best_params = []
        all_training_memberships = {}
        all_training_labels = {}
        all_training_scores = []
        all_memberships = {}
        all_labels = {}
        all_scores = []

        # this code is used for generated datasets
        self.y_cut = np.array(self.threshold(self.y, 0.5))

        for i, (train_id, test_id) in enumerate(outer_fold.split(self.X, self.y_cut)):
            print("Working on fold " + str(i + 1) + " of " + str(outer_folds))
            X_train, X_test = self.X[train_id], self.X[test_id]
            y_train, y_test = self.y[train_id], self.y[test_id]

            folds["x train " + str(i)] = X_train
            folds["y train " + str(i)] = y_train

            folds["x test " + str(i)] = X_test
            folds["y test " + str(i)] = y_test

            gs = GridSearchCV(
                pipe,
                params,
                verbose=0,
                cv=inner_folds
            )

            gs = gs.fit(X_train, y_train)
            # f = gs.best_estimator_["learning_algorithm"]

            training_memberships, training_labels, training_scores = self.evaluate_model(gs, X_train, y_train)
            all_training_memberships[i] = training_memberships
            all_training_labels[i] = training_labels
            all_training_scores.append(training_scores)

            memberships, labels, scores = self.evaluate_model(gs, X_test, y_test)

            all_memberships[i] = memberships
            all_labels[i] = labels
            all_scores.append(scores)

            best_models.append(gs.best_estimator_)
            best_params.append(gs.best_params_)

        self.folds = folds
        self.best_models = best_models
        self.best_params = best_params
        self.all_training_memberships = all_training_memberships
        self.all_training_labels = all_training_labels
        self.all_training_scores = all_training_scores
        self.all_memberships = all_memberships
        self.all_labels = all_labels
        self.all_scores = all_scores

        if self.write:
            # when the best models are found, they are serialized and stored
            self.write_model()

    def evaluate_model(self, gs, X, y):
        scores = {}
        memberships = gs.predict(X)
        predicted_labels = self.threshold(memberships)
        labels = self.threshold(y)
        TP, FP, FN, TN = self.confusion_matrix(labels, predicted_labels, memberships)
        precision = self.precision(TP, FP)
        recall = self.recall(TP, FN)
        f1 = self.f1(precision, recall)
        rmse = self.RMSE(memberships, y)

        scores["precision"] = precision
        scores["recall"] = recall
        scores["f1"] = f1
        scores["rmse"] = rmse

        return memberships, labels, scores


    def simple_split(self, dataset):
        id = dataset.iloc[:, 0].values # id
        X = dataset.iloc[:, 1].values  # x-component
        y = dataset.iloc[:, 2].values  # labels / membership
        return id, X, y

    def threshold(self, membership, n = 0.5):
        predictions = []

        for x in membership:
            if x >= n:
                predictions.append(1) # fake
            else:
                predictions.append(0) # true

        return predictions

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

    def MSE(self, prediction, y):
        return (1/len(prediction))*sum((y - prediction) ** 2)

    def RMSE(self, prediction, y):
        return np.sqrt(self.MSE(prediction, y))

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

        for i, param in enumerate(self.best_params):
            models = models.append({
                "Date": self.date, "Time": self.time, "Sample": len(self.dataset),
                "Solver": self.solver, "Params": param, "Scores": self.all_scores[i]
            }, ignore_index=True)

        models.to_csv(fullname, index=False)

        outdir = "selected_models/" + self.date + "_" + self.time + "/"
        outname = "models.pickle"

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        fullname = os.path.join(outdir, outname)
        file = open(fullname, "wb")
        dill.dump(self, file)
        file.close()

        # for i, model in enumerate(self.best_models):
        #     outdir = "selected_models/" + self.date + "_" + self.time + "/"
        #     outname = "model_(" + str(i) + ").pickle"
        #
        #     if not os.path.exists(outdir):
        #         os.makedirs(outdir)
        #
        #     fullname = os.path.join(outdir, outname)
        #
        #     file = open(fullname, "wb")
        #     pickle.dump(model, file)
        #     file.close()
