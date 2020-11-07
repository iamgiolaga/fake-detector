import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from mulearn import FuzzyInductor, fuzzifier, kernel, optimization as opt
from sklearn.model_selection import GridSearchCV, train_test_split
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

# logging configuration
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(filename='logs', filemode='w', format='%(message)s')


#plot functions
def gr_dataset():
    for lab, col, text in zip((0, 1),
                        ('blue', 'red'),
                              ('real news', 'fake news')):
        plt.scatter(X_2d[y == lab, 0],
                    X_2d[y == lab, 1],
                    label=text,
                    c=col,
                    alpha=0.5)
        plt.legend(loc="best")
        plt.title("Scatterplot")

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


# data loading
dataset = pd.read_csv("results/final_text_dataset.csv")

#dataset = dataset.head(400)

# start experiment
e = Experiment(dataset)
X, y = e.simple_split(dataset)
X = [ast.literal_eval(i) for i in X]


#reduce to 2d in order to plot
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X)

# plot data
fig = plt.figure(figsize=(10, 10))
gr_dataset()
plt.show()
fig.savefig("images/scatterplot.png")

# try a basic model, fitted with X in 2d and see how the kernel works
f = FuzzyInductor()
f.fit(X_2d, y)

fig = plt.figure(figsize=(10, 10))
gr_dataset()
gr_membership_contour(f.estimated_membership_)
plt.show()
fig.savefig("images/scatterplot_contour.png")


#split in train and test set
X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, random_state = 42)

# fit with training set
f = FuzzyInductor()

f.fit(X_train, y_train)

# predict with training set
predictions = f.predict(X_train)

# compute the root mean squared error
print("TRAINING ERROR:")
training_error = e.RMSE(predictions, y_train)
print(training_error)

# set the configuration that has been used
e.set_params(f.get_params())
e.write_experiment("training", "w")

# predict with test set
predictions = f.predict(X_test)

# compute the root mean squared error
print("TEST ERROR:")
test_error = e.RMSE(predictions, y_test)
print(test_error)

# set the configuration that has been used
e.set_params(f.get_params())
e.write_experiment("test", "w")


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
