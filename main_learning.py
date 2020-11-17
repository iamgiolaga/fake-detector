import pandas as pd
import logging
from classes.experiment import Experiment

''' DESCRIPTION '''
''' This is the file where experiments are launched '''

''' THIRD SECTION: LEARNING A MODEL TO RECOGNIZE FAKE NEWS '''

# ALL POSSIBLE VALUES FOR MY EXPERIMENTS #
PATH_TEXTS = "preprocessed_datasets/final_text_dataset.csv"
PATH_TITLES = "preprocessed_datasets/final_title_dataset.csv"
AGGREGATION_W = "word2vec"
AGGREGATION_D = "doc2vec"
GAUSSIAN_KERNEL = "gaussian"
LIN_FUZZIFIER = "linear"
EXP_FUZZIFIER = "exponential"
SOLVER_TENSORFLOW = "tensorflow"
SOLVER_GUROBI = "gurobi"

# logging configuration
#logging.getLogger().setLevel(logging.INFO)
#logging.basicConfig(filename='logs', filemode='w', format='%(message)s')

print("LEARNING")

# data loading
dataset = pd.read_csv(PATH_TEXTS)

# extract sample
dataset = dataset.head(1000)

'''
Note that when plotting a scatterplot we are dealing with 2 dimensions.
This means that a PCA with 2 components is needed whenever X has more than 2 features.
Moreover, we can consider the plotting as a special case of Experiment, since it takes the same
arguments, included the 2-components PCA.
'''

# start experiment on dataset
e = Experiment(sample = dataset,
               aggregation_mode = AGGREGATION_W,
               c = 1,
               kernel = GAUSSIAN_KERNEL,
               sigma = 1,
               fuzzifier = EXP_FUZZIFIER,
               test_size = 0.2,
               solver = SOLVER_TENSORFLOW,
               write = True,
               plot = True)
e.run_experiment()

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
