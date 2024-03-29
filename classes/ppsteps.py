import spacy
import string
import numpy as np
import re
import ast

from tqdm import tqdm
from nltk import SnowballStemmer
from sklearn.base import BaseEstimator
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.stem import PorterStemmer

''' DESCRIPTION '''
''' This file defines the set of classes that compose the preprocessing pipeline '''

class BlankRowsRemoval(BaseEstimator):
    def fit(self):
        return

    def transform(self, data):
        return data[data.map(lambda s: len(s)) > 1].reset_index()

class Lowercasing(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        tqdm.pandas()
        return data.progress_apply(lambda s: s.lower() if type(s) == str else s)

class DuplicateRowsRemoval(BaseEstimator): # removes duplicate rows
    def fit(self, data):
        return

    def transform(self, data):
        # it removes also missing values (without NaNs encoding), because they are considered duplicates as well
        try:
            return data.drop_duplicates().reset_index()
        except:
            raise TypeError # drop_duplicates() doesn't work with lists

class Tokenization(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        tqdm.pandas()
        return data.progress_apply(lambda s: [w for w in word_tokenize(s)])

class EmojiRemoval(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):

        tqdm.pandas()
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
        data = data.progress_apply(lambda s: emoji_pattern.sub(r'', str(s)))
        return data.progress_apply(lambda s: ast.literal_eval(s))

class BadCharRemoval(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        bad_characters = string.punctuation + "’" + "“" + "”" + "–" + " " + "¡" + "¿"

        tqdm.pandas()

        # remove bad characters
        data = data.progress_apply(
            lambda s: [w for w in s if not w in bad_characters and not w in "--" and not w in "..."]
        )

        return data

class NumbersRemoval(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        tqdm.pandas()
        # remove numbers
        return data.progress_apply(lambda s: [w for w in s if w.isnumeric() != True])

class RemoveWordsWithNumbers(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, data):
        return

    def transform(self, data):
        tqdm.pandas()
        # remove URLs and words that contain numbers
        return data.progress_apply(lambda s: [w for w in s if self.has_numbers(w) != True])

    def has_numbers(self, input_string):
        return any(char.isdigit() for char in input_string)

class CleaningWords(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        # remove symbols attached to words
        data = data.apply(lambda s: [re.sub(r'[^\w]', '', w) for w in s])
        bad_characters = string.punctuation + "’" + "“" + "”" + "–" + " "
        tqdm.pandas()
        data = data.progress_apply(
            lambda s: [w for w in s if not w in bad_characters and not w in "--" and not w in "..."]
        )
        return data

class DuplicateWordsRemoval(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        tqdm.pandas()
        # remove duplicate words
        return data.progress_apply(lambda s: list(dict.fromkeys(s)))

class Lemmatization(BaseEstimator):
    def fit(self, data, language):
        if language == "en":
            self.nlp = spacy.load('en_core_web_sm')
        elif language == "es":
            self.nlp = spacy.load('es')

    def transform(self, data):
        nlp = self.nlp
        tqdm.pandas()
        return data.progress_apply(lambda s: [token.lemma_ for token in nlp(s) if not token.lemma_ in "-PRON-"])

class Stemming(BaseEstimator):
    def fit(self, data, language):
        if language == "en":
            self.porter = PorterStemmer()
        elif language == "es":
            self.porter = SnowballStemmer('spanish')

    def transform(self, data):
        porter = self.porter
        tqdm.pandas()
        return data.progress_apply(lambda s: [porter.stem(w) for w in s])

class StopwordRemoval(BaseEstimator):

    # noinspection PyUnresolvedReferences
    def fit(self, data, language):
        if language == "en":
            self.stopwords = spacy.lang.en.stop_words.STOP_WORDS
        elif language == "es":
            self.stopwords = spacy.lang.es.stop_words.STOP_WORDS

    def transform(self, data):
        stopwords = self.stopwords
        tqdm.pandas()
        return data.progress_apply(lambda x: [i for i in x if not i in stopwords])

class EntityRecognition(BaseEstimator):
    def fit(self, data, language):
        if language == "en":
            self.nlp = spacy.load('en_core_web_sm')
        elif language == "es":
            self.nlp = spacy.load('es')

    def transform(self, data):
        nlp = self.nlp
        tqdm.pandas()
        return data.progress_apply(lambda s: [(i, i.label_, i.label) for i in nlp(s).ents])

class DataAugmentation(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        return

class URLRemoval(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        tqdm.pandas()
        return data.progress_apply(lambda s : [w for w in s if not "http" in w and not "www" in w])

class WordVectorization(BaseEstimator):
    def fit(self, data, language):
        if language == "en":
            self.nlp = spacy.load('en_core_web_sm')
        elif language == "es":
            self.nlp = spacy.load('es')

    def transform(self, data):
        nlp = self.nlp
        tqdm.pandas()
        return data.progress_apply(lambda s: [nlp(i).vector for i in s])

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
        tqdm.pandas()
        data = data.progress_apply(lambda s: np.nanmean(s, axis = 0))
        return data.progress_apply(lambda s: list(s))


