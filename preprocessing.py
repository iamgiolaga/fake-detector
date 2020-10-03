import string
import spacy

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

class preprocessing():

    def __init__(self, news, duplicate_removal = True, lowercasing = True, tokenization = True, noise_removal = True, lemmatization = True, stemming = False, stopword_removal = True, data_augmentation = False):
        # currently, news is a vector of strings (titles or news bodies)
        self.news = news
        self.duplicate_removal = duplicate_removal
        self.lowercasing = lowercasing
        self.tokenization = tokenization
        self.noise_removal = noise_removal
        self.lemmatization = lemmatization
        self.stemming = stemming
        self.stopword_removal = stopword_removal
        self.data_augmentation = data_augmentation

    def run_pipeline(self):
        # TODO: check combinations of operations that need to be executed together and in which order
        print("Starting preprocessing...")

        if self.duplicate_removal == True:
            self.remove_duplicates()

        if self.lowercasing == True:
            self.lowercase()

        if self.lemmatization == True:
            self.lemmatize()

        #if self.tokenization == True:
        #    self.tokenize()

        if self.noise_removal == True:
            self.remove_noise()

        if self.stemming == True: # exclusive w.r.t. lemmatization
            self.stem()

        if self.stopword_removal == True:
            self.remove_stopword()

        # TODO: Merge word pairs - Look at SpaCy's documentation

        if self.data_augmentation == True: # TODO: consider if necessary
            self.augment_data()

        print("preprocessing finished.")

        return self.news

    def get_news(self):
        return self.news

    def set_news(self, news):
        self.news = news

    def remove_duplicates(self):
        # it removes also missing values (without NaNs encoding), because they are considered duplicates as well
        print("Removing duplicates...")
        print("Items found: ", len(self.news), " rows")
        self.news = self.news.drop_duplicates()
        print("Removed items ", len(self.news), " rows")
        print("...done.")
        print("")

    def lowercase(self):
        print("Lowercasing...")
        self.news = self.news.apply(lambda s: s.lower() if type(s) == str else s)
        print("...done.")
        print("")

    def tokenize(self):
        print("Tokenization...")
        self.news = self.news.apply(lambda s: [w for w in word_tokenize(s)])
        print("...done.")
        print("")

    def hasNumbers(inputString):
        return any(char.isdigit() for char in inputString)

    def remove_noise(self):
        print("Removing noise...")
        bad_characters = string.punctuation + "’" + "“" + "”" + "‘" + "–" + " "
        # remove bad characters
        self.news = self.news.apply(
            lambda s: [w for w in s if not w in bad_characters and not w in "--" and not w in "..."]
        )

        # remove numbers
        self.news = self.news.apply(lambda s: [w for w in s if w.isnumeric() != True])

        # remove URLs and words that contain numbers
        self.news = self.news.apply(lambda s: [w for w in s if preprocessing.hasNumbers(w) != True])

        # remove words that contain '
        self.news = self.news.apply(lambda s: [w.replace("'", "") for w in s])

        print("...done.")
        print("")

    def lemmatize(self):
        print("Lemmatization...")
        nlp = spacy.load('en')
        self.news = self.news.apply(lambda s: [token.lemma_ for token in nlp(s) if not token.lemma_ in "-PRON-"])
        print("...done.")
        print("")

    def stem(self):
        print("Stemming...")
        porter = PorterStemmer()
        self.news = self.news.apply(lambda s: [porter.stem(w) for w in s])
        print("...done.")
        print("")

    def remove_stopword(self):
        print("Removing stop words...")
        #stop_words = set(stopwords.words('english'))
        spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        self.news = self.news.apply(lambda x: [i for i in x if not i in spacy_stopwords])
        print("...done.")
        print("")

    def augment_data(self):
        pass