from ppsteps import DuplicateRowsRemoval, Lowercasing, \
Tokenization, BadCharRemoval, NumbersRemoval, UrlRemoval, ApostropheRemoval, \
DuplicateWordsRemoval, Lemmatization, Stemming, StopwordRemoval,\
EntityRecognition, WordVectorization, DocVectorization, Aggregation

## DESCRIPTION ##
# This class is responsible for the preprocessing pipeline execution

class Preprocessing():

    def __init__(self, text,
                 duplicate_rows_removal = True, lowercasing = True, tokenization = True,
                 noise_removal = True, lemmatization = True, stemming = False,
                 stopword_removal = True, entity_recognition = False, data_augmentation = False,
                 word2vec = True, doc2vec = False, aggregation = True):
        # currently, text is a vector of strings (titles or news bodies)
        self.preprocessed = text
        self.lowercasing = lowercasing
        self.duplicate_rows_removal = duplicate_rows_removal
        self.tokenization = tokenization
        self.noise_removal = noise_removal
        self.lemmatization = lemmatization
        self.stemming = stemming
        self.stopword_removal = stopword_removal
        self.entity_recognition = entity_recognition
        self.data_augmentation = data_augmentation
        self.word2vec = word2vec
        self.doc2vec = doc2vec
        self.aggregation = aggregation

    def run_pipeline(self):
        # TODO: check combinations of operations that need to be executed together and in which order
        print("")
        print("Starting preprocessing...")
        print("")

        self.set_current_configuration() # stores which configuration is used

        if self.lowercasing == True:
            print("(TYPE: ", type(self.preprocessed), ")")
            self.preprocessed = self.lowercase(self.preprocessed)
            print(self.preprocessed)
            print("")

        if self.duplicate_rows_removal == True:
            print("(TYPE: ", type(self.preprocessed), ")")
            self.preprocessed = self.remove_rows_duplicates(self.preprocessed)
            print(self.preprocessed)
            print("")

        if self.entity_recognition == True:
            print("(TYPE: ", type(self.preprocessed), ")")
            self.entities = self.recognize_entity(self.preprocessed)
            print("")

        if self.lemmatization == True:
            print("(TYPE: ", type(self.preprocessed), ")")
            self.preprocessed = self.lemmatize(self.preprocessed)
            print(self.preprocessed)
            print("")

        # This is currently commented because there is a step that is also tokenizing
        # if self.tokenization == True:
        #     self.tokenize(self.preprocessed)

        if self.noise_removal == True:
            print("(TYPE: ", type(self.preprocessed), ")")
            self.preprocessed = self.remove_noise(self.preprocessed)
            print(self.preprocessed)
            print("")

        if self.stemming == True: # exclusive w.r.t. lemmatization
            print("(TYPE: ", type(self.preprocessed), ")")
            self.preprocessed = self.stem(self.preprocessed)
            print(self.preprocessed)
            print("")

        if self.stopword_removal == True:
            print("(TYPE: ", type(self.preprocessed), ")")
            self.preprocessed = self.remove_stopword(self.preprocessed)
            print(self.preprocessed)
            print("")

        # TODO: Merge word pairs - Look at SpaCy's documentation

        if self.data_augmentation == True: # TODO: consider if necessary
            self.augment_data(self.preprocessed)

        if self.word2vec == True:
            print("(TYPE: ", type(self.preprocessed), ")")
            self.wordvectors = self.wordvectorizer(self.preprocessed)

        if self.doc2vec == True:
            print("(TYPE: ", type(self.preprocessed), ")")
            self.docvectors = self.docvectorizer(self.preprocessed)

        if self.aggregation == True:
            print("(TYPE: ", type(self.preprocessed), ")")
            self.aggregated = self.aggregate(self.wordvectors)

        print("preprocessing finished.")

        return self

    def lowercase(self, data):
        print("Lowercasing...")
        l = Lowercasing()
        data = l.transform(data)
        print("...done.")
        print("")
        return data

    def remove_rows_duplicates(self, data):
        # it removes also missing values (without NaNs encoding), because they are considered duplicates as well
        print("Removing duplicates...")
        print("Items found: ", len(data), " rows")
        d = DuplicateRowsRemoval()
        data = d.transform(data)
        print("Remaining items: ", len(data), " rows")
        print("...done.")
        print("")
        return data

    def tokenize(self, data):
        print("Tokenization...")
        t = Tokenization()
        data = t.transform(data)
        print("...done.")
        print("")
        return data

    def remove_noise(self, data):
        print("Removing noise...")
        print("\t Bad characters...")
        b = BadCharRemoval()
        data = b.transform(data)
        print("\t Numbers...")
        n = NumbersRemoval()
        data = n.transform(data)
        print("\t URLs...")
        u = UrlRemoval()
        data = u.transform(data)
        print("\t Apostrophes...")
        a = ApostropheRemoval()
        data = a.transform(data)
        print("\t Duplicate words...")
        d = DuplicateWordsRemoval()
        data = d.transform(data)
        print("...done.")
        print("")
        return data

    def lemmatize(self, data):
        print("Lemmatization...")
        l = Lemmatization()
        l.fit(data) # uses nlp from spacy
        data = l.transform(data)
        print("...done.")
        print("")
        return data

    def stem(self, data):
        print("Stemming...")
        s = Stemming()
        s.fit(data) # uses stemmer from porter
        data = s.transform(data)
        print("...done.")
        print("")
        return data

    def remove_stopword(self, data):
        print("Removing stop words...")
        s = StopwordRemoval()
        s.fit(data)
        data = s.transform(data)
        print("...done.")
        print("")
        return data

    def recognize_entity(self, data): # creates a new object entities
        print("Recognizing entities...")
        e = EntityRecognition()
        e.fit(data)
        entities = e.transform(data)
        print("...done.")
        print("")
        return entities

    def wordvectorizer(self, data): # creates a new object vectors
        print("Word to vec...")
        v = WordVectorization()
        v.fit(data)
        wordvectors = v.transform(data)
        print("...done.")
        print("")
        return wordvectors

    def docvectorizer(self, data):
        print("Doc to vec...")
        d = DocVectorization()
        d.fit(data)
        docvectors = d.transform(data)
        print("...done.")
        print("")
        return docvectors

    def aggregate(self, data):
        print("Aggregating...")
        a = Aggregation()
        aggregated = a.transform(data)
        print("...done")
        print("")
        return aggregated

    def augment_data(self, data):
        pass

    def set_current_configuration(self):
        configuration = []
        if self.duplicate_rows_removal == True:
            configuration.append("Duplicate rows removal")

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

        if self.aggregation == True:
            configuration.append("Aggregation")

        self.configuration = configuration
