import pandas as pd
import logging
from classes.experiment import Experiment

# EXPERIMENT

''' DESCRIPTION '''
''' This is the file where experiments are launched on test set with the selected model '''

''' THIRD SECTION: LEARNING A MODEL TO RECOGNIZE FAKE NEWS '''

# ALL POSSIBLE VALUES FOR MY EXPERIMENTS #
PATH_TEXTS = "preprocessed_datasets/test/final_text_dataset_399.csv"
# PATH_TITLES = "preprocessed_datasets/test/final_title_dataset_1000.csv"
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

# data loading
dataset = pd.read_csv(PATH_TEXTS)

# extract sample
# dataset = dataset.head(100)

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
               plot = False)
e.run_experiment()
