import spacy
import string
import numpy as np
import re

from sklearn.base import BaseEstimator
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.stem import PorterStemmer

''' DESCRIPTION '''
''' This file defines the set of classes that compose the preprocessing pipeline '''

class Lowercasing(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        return data.apply(lambda s: s.lower() if type(s) == str else s)

class DuplicateRowsRemoval(BaseEstimator): # removes duplicate rows
    def fit(self, data):
        return

    def transform(self, data):
        # it removes also missing values (without NaNs encoding), because they are considered duplicates as well
        try:
            return data.drop_duplicates().reset_index(drop=True)
        except:
            raise TypeError # drop_duplicates() doesn't work with lists

class Tokenization(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        return data.apply(lambda s: [w for w in word_tokenize(s)])

class BadCharRemoval(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        bad_characters = string.punctuation + "’" + "“" + "”" + "–" + " "

        # remove bad characters
        data = data.apply(
            lambda s: [w for w in s if not w in bad_characters and not w in "--" and not w in "..."]
        )

        return data

class NumbersRemoval(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        # remove numbers
        return data.apply(lambda s: [w for w in s if w.isnumeric() != True])

class RemoveWordsWithNumbers(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, data):
        return

    def transform(self, data):
        # remove URLs and words that contain numbers
        return data.apply(lambda s: [w for w in s if self.has_numbers(w) != True])

    def has_numbers(self, input_string):
        return any(char.isdigit() for char in input_string)

class CleaningWords(BaseEstimator):

    def fit(self, data):
        return

    def transform(self, data):
        # remove symbols attached to words
        data = data.apply(lambda s: [re.sub(r'[^\w]', '', w) for w in s])
        bad_characters = string.punctuation + "’" + "“" + "”" + "–" + " "
        data = data.apply(
            lambda s: [w for w in s if not w in bad_characters and not w in "--" and not w in "..."]
        )
        return data

class DuplicateWordsRemoval(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        # remove duplicate words
        return data.apply(lambda s: list(dict.fromkeys(s)))

class Lemmatization(BaseEstimator):
    def fit(self, data):
        self.nlp = spacy.load('en_core_web_sm')

    def transform(self, data):
        nlp = self.nlp
        return data.apply(lambda s: [token.lemma_ for token in nlp(s) if not token.lemma_ in "-PRON-"])

class Stemming(BaseEstimator):
    def fit(self, data):
        self.porter = PorterStemmer()

    def transform(self, data):
        porter = self.porter
        return data.apply(lambda s: [porter.stem(w) for w in s])

class StopwordRemoval(BaseEstimator):

    # noinspection PyUnresolvedReferences
    def fit(self, data):
        self.stopwords = spacy.lang.en.stop_words.STOP_WORDS

    def transform(self, data):
        stopwords = self.stopwords
        return data.apply(lambda x: [i for i in x if not i in stopwords])

class EntityRecognition(BaseEstimator):
    def fit(self, data):
        self.nlp = spacy.load('en_core_web_sm')

    def transform(self, data):
        nlp = self.nlp
        return data.apply(lambda s: [(i, i.label_, i.label) for i in nlp(s).ents])

class DataAugmentation(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        return

class WordVectorization(BaseEstimator):
    def fit(self, data):
        self.nlp = spacy.load('en_core_web_sm')

    def transform(self, data):
        nlp = self.nlp
        return data.apply(lambda s: [nlp(i).vector for i in s])

class DocVectorization(BaseEstimator):
    def __init__(self, vector_size=20, window=2, min_count=1, workers=4, epochs=100):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs

    def fit(self, data):
        self.tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(data)]
        self.model = Doc2Vec(
            self.tagged_data, vector_size=self.vector_size, window=self.window,
            min_count=self.min_count, workers=self.workers, epochs=self.epochs
        )

    def transform(self, data):
        model = self.model
        return model.docvecs.most_similar(positive=[model.infer_vector(x) for x in data])

class Aggregation(BaseEstimator):
    def fit(self, data):
        return
    def transform(self, data): ## TODO: Fisher kernel aggregation
        return data.apply(lambda s : np.mean(s, axis=0))