from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class doc2vec():

    def __init__(self, text, vector_size=20, window=2, min_count=1, workers=4, epochs=100):
        self.text = text
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs

    def run_doc2vec(self):

        # Convert to Tagged
        self.prepare(self.text)

        # Train doc2vec model
        self.train()

        # Load saved doc2vec model
        model = Doc2Vec.load("test_doc2vec.model")

        # Test
        result = self.test(model)
        return result

    def prepare(self, text):
        # Convert tokenized document into gensim formated tagged data
        self.tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(text)]

    def get_tagged_data(self):
        return self.tagged_data

    def train(self):
        model = Doc2Vec(
            self.tagged_data, vector_size=self.vector_size, window=self.window,
            min_count=self.min_count, workers=self.workers, epochs=self.epochs
        )
        # Save trained doc2vec model
        model.save("test_doc2vec.model")

    def get_vocabulary(self, model):
        return model.wv.vocab

    def test(self, model):
        return model.docvecs.most_similar(positive=[model.infer_vector(x) for x in self.text])