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

''' DESCRIPTION '''
''' This file reproduces experiments of mulearn with IRIS dataset '''

source = 'https://archive.ics.uci.edu/ml/'\
         'machine-learning-databases/iris/iris.data'

iris_df = pd.read_csv(source, header=None)
iris_df.columns=['sepal_length', 'sepal_width',
                 'petal_length', 'petal_width', 'class']

iris_values = iris_df.iloc[:,0:4].values
iris_labels = iris_df.iloc[:,4].values

pca_2d = PCA(n_components=2)
iris_values_2d = pca_2d.fit_transform(iris_values)

def gr_dataset():
    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'green', 'red')):
        plt.scatter(iris_values_2d[iris_labels==lab, 0],
                    iris_values_2d[iris_labels==lab, 1],
                    label=lab,
                    c=col)

gr_dataset()
plt.show()

def to_membership_values(labels, target):
    return [1 if l==target else 0 for l in labels]

mu = {}
for target in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
    mu[target] = to_membership_values(iris_labels, target)

def gr_membership_contour(estimated_membership):
    x = np.linspace(-4, 4, 50)
    y = np.linspace(-4, 4, 50)
    X, Y = np.meshgrid(x, y)
    zs = np.array([estimated_membership((x, y))
                   for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    membership_contour = plt.contour(X, Y, Z,
                                     levels=(.1, .3, .5, .95), colors='k')
    plt.clabel(membership_contour, inline=1)

f = FuzzyInductor()
f.fit(iris_values_2d, mu['Iris-virginica'])
gr_dataset()
gr_membership_contour(f.estimated_membership_)
plt.show()
print(f.predict([[2, 0]]))

f = FuzzyInductor(fuzzifier=(fuzzifier.LinearFuzzifier, {}))
f.fit(iris_values_2d, mu['Iris-virginica'])

gr_dataset()
gr_membership_contour(f.estimated_membership_)
plt.show()

f = FuzzyInductor(fuzzifier=(fuzzifier.ExponentialFuzzifier,
                             {'profile': 'alpha', 'alpha': 0.25}))
f.fit(iris_values_2d, mu['Iris-virginica'])

gr_dataset()
gr_membership_contour(f.estimated_membership_)
plt.show()

f = FuzzyInductor(k=kernel.GaussianKernel(.3))
f.fit(iris_values_2d, mu['Iris-virginica'])

gr_dataset()
gr_membership_contour(f.estimated_membership_)
plt.show()

try:
    f = FuzzyInductor(solve_strategy=(opt.solve_optimization_gurobi, {}))
    f.fit(iris_values_2d, mu['Iris-virginica'])

    gr_dataset()
    gr_membership_contour(f.estimated_membership_)
    plt.show()
except (ModuleNotFoundError, ValueError):
    print('Gurobi not available')

f = FuzzyInductor(fuzzifier=(fuzzifier.ExponentialFuzzifier,
                             {'profile': 'alpha', 'alpha': 0.15}),
                  k=kernel.GaussianKernel(1.5),
                  solve_strategy=(opt.solve_optimization_tensorflow,
                                  {'n_iter': 20}),
                  return_profile=True)
f.fit(iris_values_2d, mu['Iris-virginica'])

gr_dataset()
gr_membership_contour(f.estimated_membership_)
plt.show()

plt.plot(f.profile_[0], mu['Iris-virginica'], '.')
plt.plot(f.profile_[1], f.profile_[2])
plt.ylim((-0.1, 1.1))
plt.show()

sigmas = [.225,.5]
parameters = {'c': [1,10,100],
              'k': [kernel.GaussianKernel(s) for s in sigmas]}

logging.getLogger('mulearn').setLevel(logging.ERROR)

f = FuzzyInductor()

with warnings.catch_warnings():
    warnings.simplefilter('ignore', FitFailedWarning)

    virginica = GridSearchCV(f, param_grid=parameters, cv=2)
    virginica.fit(iris_values_2d, mu['Iris-virginica'])

gr_dataset()
gr_membership_contour(virginica.best_estimator_.estimated_membership_)
plt.show()

saved_estimator = pickle.dumps(virginica.best_estimator_)

loaded_estimator = pickle.loads(saved_estimator)

gr_dataset()
gr_membership_contour(loaded_estimator.estimated_membership_)
plt.show()