import numpy as np
import pandas as pd
import random
import os

from tqdm import tqdm
from scipy import stats
from scipy.stats import dirichlet, poisson, binom, multinomial, zipf, uniform, norm, beta
from datetime import datetime

class Generation:

    def __init__(self, topic_1, topic_2, M, N, distribution = "uniform", write = True):
        '''
        M: # of documents (sample size)
        N: # of words per document (document size)
        '''

        self.date = datetime.now().strftime('%d.%m.%Y')
        self.time = datetime.now().strftime('%H.%M')
        self.topic_1 = topic_1
        self.topic_2 = topic_2
        self.M = M
        self.N = N
        self.distribution = distribution
        self.write = write

    def run_generation(self):

        self.topic_1 = list(dict.fromkeys(self.topic_1.columns.values[0].split(" ")).keys())
        self.topic_2 = list(dict.fromkeys(self.topic_2.columns.values[0].split(" ")).keys())

        topics = []
        topics.append(self.topic_1)
        topics.append(self.topic_2)

        corpus = pd.DataFrame()

        for i in tqdm(range(0, self.M)):

            if self.distribution == "uniform":
                p = uniform.rvs(loc = 0, scale = 1)
            elif self.distribution == "norm": # TODO: is it useful for our case, since it is (-inf, + inf)?
                p = norm.rvs()
            elif self.distribution == "beta":
                # with alfa = beta = 5 approximately normal on [0,1]
                #p = beta.rvs(5, 5)
                # with alfa = beta = 0.5 polarized on 0 and 1
                p = beta.rvs(0.5, 0.5)

            document = []
            document_pd = pd.DataFrame(columns = ["text", "membership"])
            document_pd.index.name = "index"

            for j in range(0, self.N):  # loop N words for each document
                topic_doc = binom.rvs(1, p)  # choose a topic z_n from multinomial(theta) with multinomial of 2 classes = binomial
                # w_doc = binom.rvs(len(topic_1) - 1, theta_doc[topic_doc]) # choose a word w_n from P(w_n|z_n, beta)
                N = len(self.topic_1) - 1
                x = np.arange(1, N + 1)
                a = 1.1
                weights = x ** (-a)
                weights /= weights.sum()
                bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))
                w_doc = bounded_zipf.rvs()
                # w_doc = multinomial.rvs(len(topic_1) - 1, theta_doc[topic_doc])
                word = topics[topic_doc][w_doc]  # choose a word w_n from P(w_n|z_n, beta)
                document.append(word)

            document_pd["text"] = [document]
            document_pd["membership"] = p
            corpus = corpus.append(document_pd, ignore_index = True)
        corpus.index.name = "index"

        if self.write:
            self.write_data(corpus)

    def write_data(self, data):
        outdir = "generated_text/" + self.date + "_" + self.time + "/"
        outname = "generated_dataset_" + str(len(data)) + ".csv"

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        fullname = os.path.join(outdir, outname)
        data.to_csv(fullname)



