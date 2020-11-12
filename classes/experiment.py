import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import GridSearchCV, train_test_split
from datetime import datetime
from sklearn.decomposition import PCA
from mulearn import FuzzyInductor, fuzzifier, kernel, optimization as opt

''' DESCRIPTION '''
''' This is the class that defines the class Experiment, a class
 that we use to better handle the experiments that we launch '''

class Experiment():

    def __init__(self, sample, aggregation_mode = "word2vec", c = 1, kernel = "gaussian", sigma = 1, alpha = None,
                 fuzzifier = "exponential", test_size = 0.2, solver = "tensorflow",
                 n_iter = None, pca = None, plot = False, write = False):
        self.sample = sample
        self.aggregation_mode = aggregation_mode  # word2vec or doc2vec
        self.c = c  # value for SVM
        self.kernel = kernel
        self.sigma = sigma  # value for gaussian kernel
        self.fuzzifier = fuzzifier  # linear or exponential

        if self.fuzzifier == "exponential":
            self.alpha = alpha  # decaying degree of the exponential

        self.test_size = test_size  # a percentage
        self.solver = solver  # tensorflow or gurobi
        self.n_iter = n_iter  # number of iterations for the solver
        self.pca = pca  # number of components for PCA

        if pca is not None:
            self.boolpca = True
        else:
            self.boolpca = False

        self.plot = plot  # boolean value for plotting
        self.write = write  # boolean value for writing the experiment on file

        self.datatype = sample.columns[0]  # news title or body

    def run_experiment(self):
        self.X, self.y = self.simple_split(self.sample)
        self.X = [ast.literal_eval(i) for i in self.X]

        ''' CONFIGURE FUZZY INDUCTOR '''
        # solver strategy
        if self.solver == "tensorflow":
            if self.n_iter is not None:
                self.strategy = (opt.solve_optimization_tensorflow, {'n_iter': self.n_iter})
            else:
                self.strategy = (opt.solve_optimization_tensorflow, {})
        else:
            if self.solver == "gurobi":
                if self.n_iter is not None:
                    self.strategy = (opt.solve_optimization_gurobi, {'n_iter': self.n_iter})
                else:
                    self.strategy = (opt.solve_optimization_gurobi, {})

        # fuzzifier
        if self.fuzzifier == "exponential":
            if self.alpha is not None:
                self.fuzzifier_type = (fuzzifier.ExponentialFuzzifier, {'profile': 'alpha', 'alpha': self.alpha})

            else:
                self.fuzzifier_type = (fuzzifier.ExponentialFuzzifier, {})

        else:
            if self.fuzzifier == "linear":
                self.fuzzifier_type = (fuzzifier.LinearFuzzifier, {})

        # kernel
        self.k = kernel.GaussianKernel(self.sigma)

        self.f = FuzzyInductor(solve_strategy = self.strategy,
                               fuzzifier = self.fuzzifier_type,
                               k = self.k)

        if self.plot == True:
            pca = PCA(n_components = 2)
            self.X_plot = pca.fit_transform(self.X)

            self.X_train_PCA, \
            self.X_test_PCA, \
            self.y_train_PCA, \
            self.y_test_PCA = train_test_split(self.X_plot, self.y, test_size = self.test_size, random_state = 42)

            self.f_PCA = FuzzyInductor(solve_strategy = self.strategy,
                                   fuzzifier = self.fuzzifier_type,
                                   k = self.k)

            self.f_PCA.fit(self.X_train_PCA, self.y_train_PCA)

            fig = plt.figure(figsize=(10, 10))
            self.gr_dataset(self.X_train_PCA, self.y_train_PCA, len(self.X_train_PCA))
            plt.show()
            fig.savefig("images/scatterplot_"
                        + str(len(self.X_train_PCA))
                        + "_c=" + str(self.c)
                        + "_" + self.fuzzifier
                        + "_" + str(self.k)
                        + "_" + self.solver + ".png")

            fig = plt.figure(figsize=(10, 10))
            self.gr_dataset(self.X_train_PCA, self.y_train_PCA, len(self.X_train_PCA))
            self.gr_membership_contour(self.f_PCA.estimated_membership_)
            plt.show()
            fig.savefig("images/scatterplot_countour_"
                        + str(len(self.X_train_PCA))
                        + "_c=" + str(self.c)
                        + "_" + self.fuzzifier + "fuzzy"
                        + "_" + str(self.k)
                        + "_" + self.solver + ".png")

        if self.pca is not None:
            # extract n features
            pca = PCA(n_components = self.pca)
            self.X = pca.fit_transform(self.X)

        # split in train and test set
        self.X_train, \
        self.X_test, \
        self.y_train, \
        self.y_test = train_test_split(self.X, self.y, test_size = self.test_size, random_state = 42)

        # fit with training set
        self.f.fit(self.X_train, self.y_train)

        # predict with training set (training error)
        self.predictions = self.f.predict(self.X_train)

        # compute the root mean squared error
        print("TRAINING ERROR:")
        training_error = self.RMSE(self.predictions, self.y_train)
        print(training_error)

        if self.write:
            self.write_experiment("training")

        # predict with test set (test error)
        self.predictions = self.f.predict(self.X_test)

        # compute the root mean squared error
        print("TEST ERROR:")
        test_error = self.RMSE(self.predictions, self.y_test)
        print(test_error)

        if self.write:
            self.write_experiment("test")

    def simple_split(self, dataset):
        X = dataset.iloc[:, 0].values  # x-component
        y = dataset.iloc[:, 1].values  # labels
        return X, y

    def shuffle(self, dataset):
        return dataset.sample(frac = 1, random_state = 42)

    def square_loss(self, prediction, y):
        return (y - prediction) ** 2

    def RMSE(self, prediction, y):
        loss = self.square_loss(prediction, y)
        self.score = np.sqrt(1 / len(loss) * sum(loss))
        return self.score

    def write_experiment(self, experiment_mode):

        # 1. read to see if any dataframe is already available
        # 2. if yes, read it and update it
        # 3. if not, create it and fill it

        try:
            experiments = pd.read_csv("../results/experiments.csv")
        except:
            column_names = ["Date", "Time", "Sample", "Data Type", "Aggregation Mode",
                            "PCA", "Components", "c", "Fuzzifier", "Alpha", "Kernel",
                            "Sigma", "Solver", "Iterations", "Test Size", "RMSE", "Error"]
            experiments = pd.DataFrame(columns = column_names)

        date = datetime.now().strftime('%d/%m/%Y')
        time = datetime.now().strftime('%H:%M')

        experiments = experiments.append({"Date": date, "Time": time, "Sample": len(self.sample),
                                          "Data Type": self.datatype, "Aggregation Mode": self.aggregation_mode,
                                          "PCA": self.boolpca, "Components": self.pca,
                                          "c": self.c, "Fuzzifier": self.fuzzifier,
                                          "Alpha": self.alpha, "Kernel": self.kernel,
                                          "Sigma": self.sigma, "Solver": self.solver,
                                          "Iterations": self.n_iter, "Test Size": self.test_size,
                                          "RMSE": self.score, "Error": experiment_mode}, ignore_index = True)

        experiments.to_csv("experiments.csv", index = False)

    # plot functions
    def gr_dataset(self, X, y, cardinality):
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

    def gr_membership_contour(self, estimated_membership):
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        zs = np.array([estimated_membership((x, y))
                       for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)
        membership_contour = plt.contour(X, Y, Z,
                                         levels = (.1, .3, .5, .95), colors = 'k')
        plt.clabel(membership_contour, inline = 1)