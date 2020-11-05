import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from mulearn import FuzzyInductor, fuzzifier, kernel, optimization as opt
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import FitFailedWarning
import logging
import warnings
import pickle
import ast
from experiment import Experiment

''' DESCRIPTION '''
''' This is the file where experiments are launched '''

''' THIRD SECTION: LEARNING A MODEL TO RECOGNIZE FAKE NEWS '''
print("LEARNING")
logging.getLogger().setLevel(logging.INFO)

dataset = pd.read_csv("results/final_dataset.csv")

news_values = dataset.iloc[:,0].values
news_labels = dataset.iloc[:,1].values

news_values = [ast.literal_eval(i) for i in news_values]

#apply PCA, to reduce to 2 components
pca_2d = PCA(n_components=2)
news_values_2d = pca_2d.fit_transform(news_values)

logging.basicConfig(filename='logs', filemode='w', format='%(message)s')

def gr_dataset():
    for lab, col in zip((0, 1),
                        ('blue', 'red')):
        plt.scatter(news_values_2d[news_labels==lab, 0],
                    news_values_2d[news_labels==lab, 1],
                    label=lab,
                    c=col)

gr_dataset()
plt.show()

def gr_membership_contour(estimated_membership):
    x = np.linspace(-4, 4, 50)
    y = np.linspace(-4, 4, 50)
    X, Y = np.meshgrid(x, y)
    zs = np.array([estimated_membership((x, y))
                   for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    membership_contour = plt.contour(X, Y, Z,
                                     levels=(.1, .3, .5, .95), colors='k')
    plt.clabel(membership_contour, inline=1)

f = FuzzyInductor()
f.fit(news_values_2d, news_labels)
gr_dataset()
gr_membership_contour(f.estimated_membership_)
plt.show()

#TODO: tuning c and k hyperparameters (with CV)
e = Experiment(news_values, f.get_params())
e.write_experiment()

#write_experiment("Sample = "+str(len(news_values))+", Exponential fuzzifier, Gaussian kernel - Sigma = 1")

