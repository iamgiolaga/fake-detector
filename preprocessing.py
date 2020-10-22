from ppsteps import DuplicateRemoval, Lowercasing, \
Tokenization, NoiseRemoval, Lemmatization, Stemming, StopwordRemoval,\
EntityRecognition, DataAugmentation, WordVectorization, DocVectorization

## DESCRIPTION ##
# This class is responsible for the preprocessing pipeline execution

class Preprocessing():

    def __init__(self, text,
                 duplicate_removal = True, lowercasing = True, tokenization = True,
                 noise_removal = True, lemmatization = True, stemming = False,
                 stopword_removal = True, entity_recognition = False, data_augmentation = False,
                 word2vec = True, doc2vec = False):
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
        self.word2vec = word2vec
        self.doc2vec = doc2vec

    def run_pipeline(self):
        # TODO: check combinations of operations that need to be executed together and in which order
        print("Starting preprocessing...")

        self.set_current_configuration() # stores which configuration is used

        if self.duplicate_removal == True:
            self.remove_duplicates()

        if self.lowercasing == True:
            self.lowercase()

        if self.entity_recognition == True:
            self.recognize_entity()

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

        if self.word2vec == True:
            self.wordvectorizer()

        if self.doc2vec == True:
            self.docvectorizer()

        print("preprocessing finished.")

        return self

    def remove_duplicates(self):
        # it removes also missing values (without NaNs encoding), because they are considered duplicates as well
        print("Removing duplicates...")
        print("Items found: ", len(self.preprocessed), " rows")
        d = DuplicateRemoval()
        self.preprocessed = d.transform(self.preprocessed)
        print("Remaining items: ", len(self.preprocessed), " rows")
        print("...done.")
        print("")

    def lowercase(self):
        print("Lowercasing...")
        l = Lowercasing()
        self.preprocessed = l.transform(self.preprocessed)
        print("...done.")
        print("")

    def tokenize(self):
        print("Tokenization...")
        t = Tokenization()
        self.preprocessed = t.transform(self.preprocessed)
        print("...done.")
        print("")

    def remove_noise(self):
        print("Removing noise...")
        n = NoiseRemoval()
        self.preprocessed = n.transform(self.preprocessed)
        print("...done.")
        print("")

    def lemmatize(self):
        print("Lemmatization...")
        l = Lemmatization()
        l.fit(self.preprocessed) # uses nlp from spacy
        self.preprocessed = l.transform(self.preprocessed)
        print("...done.")
        print("")

    def stem(self):
        print("Stemming...")
        s = Stemming()
        s.fit(self.preprocessed) # uses stemmer from porter
        self.preprocessed = s.transform(self.preprocessed)
        print("...done.")
        print("")

    def remove_stopword(self):
        print("Removing stop words...")
        s = StopwordRemoval()
        s.fit(self.preprocessed)
        self.preprocessed = s.transform(self.preprocessed)
        print("...done.")
        print("")

    def recognize_entity(self): # creates a new object entities
        print("Recognizing entities...")
        e = EntityRecognition()
        e.fit(self.preprocessed)
        self.entities = e.transform(self.preprocessed)
        print("...done.")
        print("")

    def wordvectorizer(self): # creates a new object vectors
        print("Word to vec...")
        v = WordVectorization()
        v.fit(self.preprocessed)
        self.wordvectors = v.transform(self.preprocessed)
        print("...done.")
        print("")

    def docvectorizer(self):
        print("Doc to vec...")
        d = DocVectorization()
        d.fit()
        self.docvectors = d.transform(self.preprocessed)
        print("...done.")
        print("")

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

        if self.word2vec == True:
            configuration.append("Word to vec")

        if self.doc2vec == True:
            configuration.append("Doc to vec")

        self.configuration = configuration
