import spacy
import string

from abc import ABC, abstractmethod
from nltk.tokenize import word_tokenize
from spacy import displacy
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

class Ppstep(ABC): # abstract class which describes the general structure of a preprocessing step
    @abstractmethod # implementing fit and transform methods to be sklearn-compliant
    def fit(self, data):
        pass
    def transform(self, data):
        pass

class DuplicateRemoval(Ppstep):
    def fit(self, data):
        return

    def transform(self, data):
        # it removes also missing values (without NaNs encoding), because they are considered duplicates as well
        return data.drop_duplicates()

class Lowercasing(Ppstep):
    def fit(self, data):
        return

    def transform(self, data):
        return data.apply(lambda s: s.lower() if type(s) == str else s)

class Tokenization(Ppstep):
    def fit(self, data):
        return

    def transform(self, data):
        return data.apply(lambda s: [w for w in word_tokenize(s)])

class NoiseRemoval(Ppstep):
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

class Lemmatization(Ppstep):
    def fit(self, data):
        self.nlp = spacy.load('en')

    def transform(self, data):
        nlp = self.nlp
        return data.apply(lambda s: [token.lemma_ for token in nlp(s) if not token.lemma_ in "-PRON-"])

class Stemming(Ppstep):
    def fit(self, data):
        self.porter = PorterStemmer()

    def transform(self, data):
        porter = self.porter
        return data.apply(lambda s: [porter.stem(w) for w in s])

class StopwordRemoval(Ppstep):
    def fit(self, data):
        self.stopwords = spacy.lang.en.stop_words.STOP_WORDS

    def transform(self, data):
        stopwords = self.stopwords
        return data.apply(lambda x: [i for i in x if not i in stopwords])

class EntityRecognition(Ppstep):
    def fit(self, data):
        self.nlp = spacy.load('en')

    def transform(self, data):
        nlp = self.nlp
        return data.apply(lambda s: [(i, i.label_, i.label) for i in nlp(s).ents])

class DataAugmentation(Ppstep):
    def fit(self, data):
        return

    def transform(self, data):
        return

class Vectorization(Ppstep):
    def fit(self, data):
        self.nlp = spacy.load('en_core_web_sm')

    def transform(self, data):
        nlp = self.nlp
        return data.apply(lambda s: [[j.vector for j in nlp(i)] for i in s])