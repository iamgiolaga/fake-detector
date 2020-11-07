import numpy as np
from datetime import datetime

''' DESCRIPTION '''
''' This is the class that defines the class Experiment, a class
 that we use to better handle the experiments that we launch '''

class Experiment():

    def __init__(self, sample, params = None):
        self.sample = sample
        self.params = params
        self.datatype = sample.columns[0] # news title or body
        self.blankline = True

        if params is not None:
            self.params = params

    def simple_split(self, dataset):
        X = dataset.iloc[:, 0].values # x-component
        y = dataset.iloc[:, 1].values # labels
        return X, y

    def shuffle(self, dataset):
        return dataset.sample(frac=1, random_state=42)

    def square_loss(self, prediction, y):
        return (y - prediction) ** 2

    def RMSE(self, prediction, y):
        loss = self.square_loss(prediction, y)
        self.score = np.sqrt(1 / len(loss) * sum(loss))
        return self.score

    def write_experiment(self, experiment_mode = "training", aggregation_mode = "w"):
        self.extract_params()

        configuration = "Sample = " + str(len(self.sample)) \
                        + " (" + self.datatype + ")" \
                        + ", c = " + str(self.c) \
                        + ", Fuzzifier = " + str(self.fuzzifier) \
                        + ", Kernel = " + str(self.k) \
                        + ", Solver = " + str(self.solve_strategy)

        now = datetime.now().strftime('%d/%m/%Y - %H:%M')

        f = ""
        message = ""

        if aggregation_mode == "w":
            f = open("w2vec_experiments.txt", "a")

        else:
            if aggregation_mode == "d":
                f = open("d2vec_experiments.txt", "a")

        if experiment_mode == "training":
            message = now + " | " + str(configuration) + " | Training error = " + str(self.score)

        else:
            if experiment_mode == "test":
                message = now + " | " + str(configuration) + " | Test error = " + str(self.score)

        if self.blankline == True:
            f.write("\n" + message + "\n")

        else:
            f.write(message + "\n")

        f.close()
        self.blankline = not self.blankline

    def set_params(self, params):
        self.params = params

    def extract_params(self):
        self.c = self.params["c"]
        self.fuzzifier = self.params["fuzzifier"][0].__name__
        self.k = self.params["k"]
        self.solve_strategy = self.params["solve_strategy"][0].__name__