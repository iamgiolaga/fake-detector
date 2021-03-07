import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

from mulearn import optimization as opt
from gurobipy.gurobipy import GurobiError
from sklearn.decomposition import PCA
from datetime import datetime

date = datetime.now().strftime('%d.%m.%Y')
time = datetime.now().strftime('%H:%M')

''' DESCRIPTION '''
''' This is the file dedicated to plot representations '''

# plot functions
def gr_dataset(X, y, cardinality):
    for lab, col, text in zip((0, 1),
                              ('blue', 'red'),
                              ('real news', 'fake news')):
        plt.title("Scatterplot with " + str(cardinality) + " data points")
        plt.scatter(X[y == lab, 0],
                    X[y == lab, 1],
                    label = text,
                    c = col,
                    alpha = 0.5)
        plt.legend(loc="best")

def gr_membership_contour(estimated_membership):
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)

    zs = np.array([estimated_membership((x, y))
                   for x, y in zip(np.ravel(X), np.ravel(Y))])

    Z = zs.reshape(X.shape)
    membership_contour = plt.contour(X, Y, Z,
                                     levels = (.1, .3, .5, .95), colors = 'k')
    plt.clabel(membership_contour, inline = 1)

fullname = "selected_models/08.02.2021_11.55/models.pickle"
file = open(fullname, "rb")
models = pickle.load(file)
all_memberships = models.all_memberships
folds = models.folds

for i in range(0,5):
    X_train = models.folds['x train '+str(i)]
    y_train = models.folds['y train '+str(i)]
    X_test = models.folds['x test '+str(i)]
    y_test = models.folds['y test '+str(i)]

    pca = PCA(n_components = 2)
    X_train_2d = pca.fit_transform(X_train)

    model = models.best_models[i]["learning_algorithm"]

    try:
        model.fit(X_train_2d, y_train)
    except GurobiError as exception:
        print("Adjusting...")
        result = str(exception).split("adjustment of ")
        result = result[1].split(" would be")
        adjustment = float(result[0])
        model.solve_strategy = (opt.solve_optimization_gurobi, {'adjustment': adjustment})
        model.fit(X_train_2d, y_train)

    X_test_2d = pca.fit_transform(X_test)

    fig = plt.figure(figsize = (10, 10))
    gr_dataset(X_test_2d, y_test, len(X_test_2d))
    gr_membership_contour(model.estimated_membership_)
    plt.show()

    outdir = "img/"
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    outname = "scatterplot_countour_" + date + "_" + time + "(" + str(i) + ").png"
    fullname = os.path.join(outdir, outname)
    fig.savefig(fullname)