import numpy as np
from datetime import datetime

''' DESCRIPTION '''
''' This is the class that defines the class Experiment, a class
 that we use to better handle the experiments that we launch '''

class Experiment():

    def __init__(self, sample, params = None):
        self.sample = sample
        self.params = params

        self.c = self.params["c"]
        self.fuzzifier = self.params["fuzzifier"][0]
        self.k = self.params["k"]
        self.solve_strategy = self.params["solve_strategy"][0]

    def simple_split(self, dataset):
        X = dataset.iloc[:, 0].values # x-component
        y = dataset.iloc[:, 1].values # labels
        return X, y

    def shuffle(dataset):
        return dataset.sample(frac=1, random_state=42)

    def square_loss(self, prediction, y):
        return (y - prediction) ** 2

    def RMSE(self, prediction, y):
        loss = self.square_loss(prediction, y)
        return np.sqrt(1 / len(loss) * sum(loss))

    def write_experiment(self):
        message = "Sample = " + str(len(self.sample)) \
                  + ", c = " + str(self.c) \
                  + ", Fuzzifier = " + str(self.fuzzifier) \
                  + ", Kernel = " + str(self.k) \
                  + ", Solver = " + str(self.solve_strategy)
        now = datetime.now().strftime('%d/%m/%Y - %H:%M')
        f = open("experiments.txt", "a")
        f.write(now + " - " + str(message) + "\n")
        f.close()

    def set_params(self, params):
        self.params = params