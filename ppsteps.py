import spacy
import string

from sklearn.base import BaseEstimator
from nltk.tokenize import word_tokenize
from spacy import displacy
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

class DuplicateRemoval(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        # it removes also missing values (without NaNs encoding), because they are considered duplicates as well
        return data.drop_duplicates()

class Lowercasing(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        return data.apply(lambda s: s.lower() if type(s) == str else s)

class Tokenization(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        return data.apply(lambda s: [w for w in word_tokenize(s)])

class NoiseRemoval(BaseEstimator):
    def fit(self, data):
        return

    def transform(self, data):
        bad_characters = string.punctuation + "’" + "“" + "”" + "‘" + "–" + " "

        # remove bad characters
        data = data.apply(
            lambda s: [w for w in s if not w in bad_characters and not w in "--" and not w in "..."]
        )

        # remove numbers
        data = data.apply(lambda s: [w for w in s if w.isnumeric() != True])

        # remove URLs and words that contain numbers
        data = data.apply(lambda s: [w for w in s if NoiseRemoval.hasNumbers(w) != True])

        # remove words that contain '
        data = data.apply(lambda s: [w.replace("'", "") for w in s])

        return data

    def hasNumbers(inputString):
        return any(char.isdigit() for char in inputString)

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

class Vectorization(BaseEstimator):
    def fit(self, data):
        self.nlp = spacy.load('en_core_web_sm')

    def transform(self, data):
        nlp = self.nlp
        return data.apply(lambda s: [[j.vector for j in nlp(i)] for i in s])