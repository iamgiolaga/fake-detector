import numpy as np
import pandas as pd
import random
import os

from scipy import stats
from scipy.stats import dirichlet, poisson, binom, multinomial, zipf
from datetime import datetime
from random import sample

from classes.generation import Generation

''' DESCRIPTION '''
''' This is the file where experiments are artificial documents are generated through LDA '''

# K = 2 # number of topics
M = 1000 # number of documents
# V # number of unique words, part of the vocabulary
#alpha = np.array([1, 10]) # parameter of the prior Dirichlet for the document-topic distribution, has length k
# beta # parameter of the prior Dirichlet for the topic-word distribution
N = 1000 # number of words for each document / choose N from Poisson(eps)

topic_1 = pd.read_csv("topic_samples/topic_1_mixed.csv")
topic_2 = pd.read_csv("topic_samples/topic_2_mixed.csv")

# topic_1 = list(dict.fromkeys(topic_1.columns.values[0].split(" ")).keys())
# topic_2 = list(dict.fromkeys(topic_2.columns.values[0].split(" ")).keys())
#
# topics = []
# topics.append(topic_1)
# topics.append(topic_2)
#
# print(topic_1)
# print(topic_2)
# print("")

lda = Generation(topic_1, topic_2, M, N, distribution="beta")
lda.run_generation()

# date = datetime.now().strftime('%d.%m.%Y')
# time = datetime.now().strftime('%H.%M')
# quantiles = np.array([0.2, 0.2, 0.6])  # specify quantiles

# for i in range(1,11):
#     corpus = []
#     alpha = np.array([i, 11 - i])
#     theta_doc = dirichlet.rvs(alpha)[0]  # choose theta from Dir(a)
#     for doc in range(0, M): # loop M documents
#         #alpha = random.sample([1, 2, 3, 4, 5, 6, 7], 2)
#         document = []
#
#         for j in range(0, N): # loop N words for each document
#             topic_doc = binom.rvs(1, theta_doc[0]) # choose a topic z_n from multinomial(theta) with multinomial of 2 classes = binomial
#             #w_doc = binom.rvs(len(topic_1) - 1, theta_doc[topic_doc]) # choose a word w_n from P(w_n|z_n, beta)
#             N = len(topic_1) - 1
#             x = np.arange(1, N + 1)
#             a = 1.1
#             weights = x ** (-a)
#             weights /= weights.sum()
#             bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))
#             w_doc = bounded_zipf.rvs()
#             #w_doc = multinomial.rvs(len(topic_1) - 1, theta_doc[topic_doc])
#             print(w_doc)
#             word = topics[topic_doc][w_doc] # choose a word w_n from P(w_n|z_n, beta)
#             document.append(word)
#
#         corpus.append(document)
#
#     # for corp in corpus:
#     #    print(' '.join(corp))
#
#     corpus = pd.DataFrame(corpus)
#
#     outdir = "generated_text/" + date + "_" + time + "/"
#     outname = "membership_" + str(theta_doc[0]) + ".csv"
#
#     if not os.path.exists(outdir):
#         os.makedirs(outdir)
#
#     fullname = os.path.join(outdir, outname)
#     corpus.to_csv(fullname)


# for each of the N words w_n
    # choose a topic z_n from multinomial(theta)
    # choose a word w_n from P(w_n|z_n, beta)