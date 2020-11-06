from datetime import datetime

''' DESCRIPTION '''
''' This is the class that defines the class Experiment, a class
 that we use to better handle the experiments that we launch '''


class Experiment():

    def __init__(self, sample, params):
        self.sample = sample
        self.params = params

        self.c = self.params["c"]
        self.fuzzifier = self.params["fuzzifier"][0]
        self.k = self.params["k"]
        self.solve_strategy = self.params["solve_strategy"][0]

    def write_experiment(self):
        message = "Sample = " + str(len(self.sample)) \
                  , " c = " + str(self.c) \
                  , " " + str(self.fuzzifier) \
                  , " " + str(self.k) \
                  , " solver = " + str(self.solve_strategy)
        now = datetime.now().strftime('%d/%m/%Y - %H:%M')
        f = open("experiments.txt", "a")
        f.write(now + " - " + str(message) + "\n")
        f.close()