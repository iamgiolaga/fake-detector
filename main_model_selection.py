import pandas as pd
from mulearn import FuzzyInductor
from sklearn.pipeline import Pipeline
from classes.model import Model

# MODEL SELECTION

''' DESCRIPTION '''
''' This is the file where models are selected by using the training set '''

# TODO: Here i use Pipe objects from Sklearn in order to select the best model
pipe = Pipeline()

# Whenever a model is selected, an instance of Model is created and then written
f = FuzzyInductor()
m = Model(f)
m.write_model()