import string
import spacy

from nltk.tokenize import word_tokenize
from spacy import displacy
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

class preprocessing():

    def __init__(self, text,
                 duplicate_removal = True, lowercasing = True, tokenization = True,
                 noise_removal = True, lemmatization = True, stemming = False,
                 stopword_removal = True, entity_recognition = False, data_augmentation = False,
                 word_vector = True):
        # currently, text is a vector of strings (titles or news bodies)
        self.preprocessed = text
        self.duplicate_removal = duplicate_removal
        self.lowercasing = lowercasing
        self.tokenization = tokenization
        self.noise_removal = noise_removal
        self.lemmatization = lemmatization
        self.stemming = stemming
        self.stopword_removal = stopword_removal
        self.entity_recognition = entity_recognition
        self.data_augmentation = data_augmentation
        self.word_vector = word_vector

    def run_pipeline(self):
        # TODO: check combinations of operations that need to be executed together and in which order
        print("Starting preprocessing...")

        self.set_current_configuration() # stores which configuration is used

        if self.duplicate_removal == True:
            self.remove_duplicates()

        if self.lowercasing == True:
            self.lowercase()

        if self.entity_recognition == True:
            self.entities = self.recognize_entity()

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

        if self.word_vector == True:
            self.vectors = self.vectorize()

        print("preprocessing finished.")

        return self

    def remove_duplicates(self):
        # it removes also missing values (without NaNs encoding), because they are considered duplicates as well
        print("Removing duplicates...")
        print("Items found: ", len(self.preprocessed), " rows")
        self.preprocessed = self.preprocessed.drop_duplicates()
        print("Removed items ", len(self.preprocessed), " rows")
        print("...done.")
        print("")

    def lowercase(self):
        print("Lowercasing...")
        self.preprocessed = self.preprocessed.apply(lambda s: s.lower() if type(s) == str else s)
        print("...done.")
        print("")

    def tokenize(self):
        print("Tokenization...")
        self.preprocessed = self.preprocessed.apply(lambda s: [w for w in word_tokenize(s)])
        print("...done.")
        print("")

    def hasNumbers(inputString):
        return any(char.isdigit() for char in inputString)

    def remove_noise(self):
        print("Removing noise...")
        bad_characters = string.punctuation + "’" + "“" + "”" + "‘" + "–" + " "
        # remove bad characters
        self.preprocessed = self.preprocessed.apply(
            lambda s: [w for w in s if not w in bad_characters and not w in "--" and not w in "..."]
        )

        # remove numbers
        self.preprocessed = self.preprocessed.apply(lambda s: [w for w in s if w.isnumeric() != True])

        # remove URLs and words that contain numbers
        self.preprocessed = self.preprocessed.apply(lambda s: [w for w in s if preprocessing.hasNumbers(w) != True])

        # remove words that contain '
        self.preprocessed = self.preprocessed.apply(lambda s: [w.replace("'", "") for w in s])

        print("...done.")
        print("")

    def lemmatize(self):
        print("Lemmatization...")
        nlp = spacy.load('en')
        self.preprocessed = self.preprocessed.apply(lambda s: [token.lemma_ for token in nlp(s) if not token.lemma_ in "-PRON-"])
        print("...done.")
        print("")

    def stem(self):
        print("Stemming...")
        porter = PorterStemmer()
        self.preprocessed = self.preprocessed.apply(lambda s: [porter.stem(w) for w in s])
        print("...done.")
        print("")

    def remove_stopword(self):
        print("Removing stop words...")
        #stop_words = set(stopwords.words('english'))
        spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        self.preprocessed = self.preprocessed.apply(lambda x: [i for i in x if not i in spacy_stopwords])
        print("...done.")
        print("")

    def recognize_entity(self):
        print("Recognizing entities...")
        nlp = spacy.load('en')
        entities = self.preprocessed.apply(lambda s: [(i, i.label_, i.label) for i in nlp(s).ents])
        print("...done.")
        print("")
        return entities

    def vectorize(self):
        print("Vectorizing...")
        nlp = spacy.load('en_core_web_sm')
        vectors = self.preprocessed.apply(lambda s: [[j.vector for j in nlp(i)] for i in s])
        print("...done.")
        print("")
        return vectors

    def augment_data(self):
        pass

    def set_current_configuration(self):
        configuration = []
        if self.duplicate_removal == True:
            configuration.append("Duplicate removal")

        if self.lowercasing == True:
            configuration.append("Lowercasing")

        if self.entity_recognition == True:
            configuration.append("Entity recognition")

        if self.lemmatization == True:
            configuration.append("Lemmatization")

        # if self.tokenization == True:
        #    configuration.append("Tokenization")

        if self.noise_removal == True:
            configuration.append("Noise removal")

        if self.stemming == True:  # exclusive w.r.t. lemmatization
            configuration.append("Stemming")

        if self.stopword_removal == True:
            configuration.append("Stop words removal")

        if self.data_augmentation == True:  #
            configuration.append("Data augmentation")

        if self.word_vector == True:
            configuration.append("Word vector")

        self.configuration = configuration

    def get_preprocessed(self):
        return self.preprocessed

    def get_entities(self):
        return self.entities
