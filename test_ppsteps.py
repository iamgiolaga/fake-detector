import unittest
import pandas as pd

from classes.ppsteps import DuplicateRowsRemoval, BadCharRemoval, DuplicateWordsRemoval, Lowercasing, Lemmatization, \
    NumbersRemoval, RemoveWordsWithNumbers, CleaningWords, Stemming, StopwordRemoval, Aggregation

''' DESCRIPTION '''
''' This file defines the unit testing'''

class TestLowercasing(unittest.TestCase): # for LOWERCASING
    # input should be a string

    def test_single_row(self):
        example = pd.Series(["Donald DoNald tRump"])
        l = Lowercasing()
        result = l.transform(example)
        expected_result = pd.Series(["donald donald trump"])
        pd.testing.assert_series_equal(expected_result, result)

    def test_multi_row(self):
        example = pd.Series(["Donald DoNald tRump", "thiS iS AnOther wrong STRING"])
        l = Lowercasing()
        result = l.transform(example)
        expected_result = pd.Series(["donald donald trump", "this is another wrong string"])
        pd.testing.assert_series_equal(expected_result, result)

class TestDuplicateRows(unittest.TestCase): # for DUPLICATE ROWS REMOVAL

    def test_duplicate_string_rows(self):
        example = pd.Series(["this is the first news", "this is the second news",
                             "this is the second news", "this is the first news", "this is the last"]
                            )
        d = DuplicateRowsRemoval()
        result = d.transform(example)
        expected_result = pd.Series(["this is the first news", "this is the second news", "this is the last"])
        pd.testing.assert_series_equal(expected_result, result)

    def test_duplicate_vectors_rows(self):
        example = pd.Series([["this", "is", "the", "first"], ["this", "is", "the", "first"], ["final", "string"]])
        d = DuplicateRowsRemoval()
        #result = d.transform(example)
        expected_result = pd.Series([["this", "is", "the", "first"], ["final", "string"]])
        with self.assertRaises(TypeError):
            d.transform(example)

class TestLemmatization(unittest.TestCase): # for LEMMATIZATION

    def test_single_row(self):
        example = pd.Series(["donald trump was elected president in 2017. americans can't believe it."])
        l = Lemmatization()
        l.fit(example)
        result = l.transform(example)
        expected_result = pd.Series([["donald", "trump", "be", "elect", "president", "in", "2017", ".",
                                      "americans", "can", "not", "believe", "."]])
        pd.testing.assert_series_equal(expected_result, result)

    def test_multi_row(self):
        example = pd.Series(["donald trump was elected president in 2017. americans can't believe it.",
                            "www.wikipedia.org is the 1st website that people would use to search information"])
        l = Lemmatization()
        l.fit(example)
        result = l.transform(example)
        expected_result = pd.Series([["donald", "trump", "be", "elect", "president", "in", "2017", ".",
                                      "americans", "can", "not", "believe", "."],
                                     ["www.wikipedia.org", "be", "the", "1st", "website", "that",
                                      "people", "would", "use", "to", "search", "information"]])
        pd.testing.assert_series_equal(expected_result, result)

class TestBadCharRemoval(unittest.TestCase): # for NOISE REMOVAL

    def test_single_row(self):
        example = pd.Series([["“", "100%", "we", "can", "'t", "do", "this", "”", "-", "this", "were" ,"the",
                              "words", "of", "Mr.", "Brown..."]])
        b = BadCharRemoval()
        result = b.transform(example)
        expected_result = pd.Series([["100%", "we", "can", "'t", "do", "this", "this", "were", "the",
                                      "words", "of", "Mr.", "Brown..."]])
        pd.testing.assert_series_equal(expected_result, result)

    def test_multiple_row(self):
        example = pd.Series([["“", "100%", "we", "can", "'t", "do", "this", "”", "-", "this", "were", "the",
                              "words", "of", "Mr.", "Brown..."],
                             ["he", "won't", "tell", "another ", "word!", "", "!"]])
        b = BadCharRemoval()
        result = b.transform(example)
        expected_result = pd.Series([["100%", "we", "can", "'t", "do", "this", "this", "were", "the",
                                      "words", "of", "Mr.", "Brown..."],
                                     ["he", "won't", "tell", "another ", "word!"]])
        pd.testing.assert_series_equal(expected_result, result)

class TestNumbersRemoval(unittest.TestCase): # for NOISE REMOVAL

    def test_single_row(self):
        example = pd.Series([["w3schools", "is", "a", "useful", "website",
                              "with", "more", "than", "500", "pages"]])
        n = NumbersRemoval()
        result = n.transform(example)
        expected_result = pd.Series([["w3schools", "is", "a", "useful", "website",
                                      "with", "more", "than", "pages"]])
        pd.testing.assert_series_equal(expected_result, result)

    def test_multiple_row(self):
        example = pd.Series([["w3schools", "is", "a", "useful", "website",
                              "with", "more", "than", "500", "pages"],
                             ["30000", "people", "were", "protesting"]])
        n = NumbersRemoval()
        result = n.transform(example)
        expected_result = pd.Series([["w3schools", "is", "a", "useful", "website",
                                      "with", "more", "than", "pages"],
                                     ["people", "were", "protesting"]])
        pd.testing.assert_series_equal(expected_result, result)

class TestWordsWithNumbersRemoval(unittest.TestCase): # for NOISE REMOVAL

    def test_single_row(self):
        example = pd.Series([["www.wikipedia.org", "is", "the", "1st", "website", "that", "people", "would",
                              "use", "to", "search", "information"]])
        u = RemoveWordsWithNumbers()
        result = u.transform(example)
        expected_result = pd.Series([["www.wikipedia.org", "is", "the", "website", "that", "people", "would",
                              "use", "to", "search", "information"]])
        pd.testing.assert_series_equal(expected_result, result)

    def test_multiple_row(self):
        example = pd.Series([["www.wikipedia.org", "is", "the", "1st", "website", "that", "people", "would",
                              "use", "to", "search", "information"],
                             ["33", "is", "my3", "lucky", "number"]])
        u = RemoveWordsWithNumbers()
        result = u.transform(example)
        expected_result = pd.Series([["www.wikipedia.org", "is", "the", "website", "that", "people", "would",
                                      "use", "to", "search", "information"],
                                     ["is", "lucky", "number"]])
        pd.testing.assert_series_equal(expected_result, result)

class TestWordsCleaning(unittest.TestCase): # for NOISE REMOVAL

    def test_single_row(self):
        example = pd.Series([["“", "100%", "we", "can", "'t", "do", "this", "”", "-", "this", "were", "the",
                              "words", "of", "Mr.", "Brown..."]])
        c = CleaningWords()
        result = c.transform(example)
        expected_result = pd.Series([["100", "we", "can", "t", "do", "this", "this", "were", "the",
                                      "words", "of", "Mr", "Brown"]])
        pd.testing.assert_series_equal(expected_result, result)

    def test_multiple_row(self):
        example = pd.Series([["“", "100%", "we", "can", "'t", "do", "this", "”", "-", "this", "were", "the",
                              "words", "of", "Mr.", "Brown..."],
                             ["∞", "^", "§", "M&m's", "is", "a", " ", "  ", "known", "brand"]])
        c = CleaningWords()
        result = c.transform(example)
        expected_result = pd.Series([["100", "we", "can", "t", "do", "this", "this", "were", "the",
                                      "words", "of", "Mr", "Brown"],
                                     ["Mms", "is", "a", "known", "brand"]])
        pd.testing.assert_series_equal(expected_result, result)

class TestDuplicateWordsRemoval(unittest.TestCase): # for NOISE REMOVAL
    # input should be a list of strings

    def test_single_row(self):
        example = pd.Series([["donald", "donald", "trump"]])
        d = DuplicateWordsRemoval()
        result = d.transform(example)
        expected_result = pd.Series([["donald", "trump"]])
        pd.testing.assert_series_equal(expected_result, result)

    def test_multi_row(self):
        example = pd.Series([["donald", "donald", "trump"], ["this", "is", "is", "a", "duplicate", "word", "word"]])
        d = DuplicateWordsRemoval()
        result = d.transform(example)
        expected_result = pd.Series([["donald", "trump"], ["this", "is", "a", "duplicate", "word"]])
        pd.testing.assert_series_equal(expected_result, result)

class TestStemming(unittest.TestCase): # for STEMMING

    def test_single_row(self):
        example = pd.Series([["she", "studied", "a", "lot", "and", "she",
                              "is", "ready", "to", "take", "the", "exam"]])
        s = Stemming()
        s.fit(example)
        result = s.transform(example)
        expected_result = pd.Series([["she", "studi", "a", "lot", "and", "she", "is",
                                      "readi", "to", "take", "the", "exam"]])
        pd.testing.assert_series_equal(expected_result, result)

    def test_multi_row(self):
        example = pd.Series([["she", "studied", "a", "lot", "and", "she",
                              "is", "ready", "to", "take", "the", "exam"],
                             ["we", "have", "to", "decide", "as", "soon", "as", "possible"]])
        s = Stemming()
        s.fit(example)
        result = s.transform(example)
        expected_result = pd.Series([["she", "studi", "a", "lot", "and", "she", "is",
                                      "readi", "to", "take", "the", "exam"],
                                     ["we", "have", "to", "decid", "as", "soon", "as", "possibl"]])
        pd.testing.assert_series_equal(expected_result, result)

class TestStopWordRemoval(unittest.TestCase): # for STOP WORD REMOVAL

    def test_single_row(self):
        example = pd.Series([["she", "studied", "a", "lot", "and", "she",
                              "is", "ready", "to", "take", "the", "exam"]])
        s = StopwordRemoval()
        s.fit(example)
        result = s.transform(example)
        expected_result = pd.Series([["studied", "lot", "ready", "exam"]])
        pd.testing.assert_series_equal(expected_result, result)

    def test_multi_row(self):
        example = pd.Series([["she", "studied", "a", "lot", "and", "she",
                              "is", "ready", "to", "take", "the", "exam"],
                             ["we", "have", "to", "decide", "as", "soon", "as", "possible"]])
        s = StopwordRemoval()
        s.fit(example)
        result = s.transform(example)
        expected_result = pd.Series([["studied", "lot", "ready", "exam"],
                                     ["decide", "soon", "possible"]])
        pd.testing.assert_series_equal(expected_result, result)

## HOW CAN I TEST WORD2VEC, DOC2VEC? ##

class TestAggregation(unittest.TestCase):
    '''  two solutions: '''
    '''
    method 1: aggregate m vectors from word2vec and compute for each vector its mean w.r.t. k features, 
    with this method we obtain a vector of m means, so a vector of m length (variable for each document)
    method 2: (the one we want) aggregate m vectors in one vector of k length computing the mean w.r.t m vectors,
    with this method we obtain a vector of k means, so a vector of k length (fixed for each document)
    '''

    def test_single_row_aggregation(self):
        w = pd.Series([[[2,4,6],
                        [1,2,3],
                        [3,3,3]]])

        a = Aggregation()
        result = a.transform(w)
        expected_result = pd.Series([[2, 3, 4]])
        pd.testing.assert_series_equal(expected_result, result)

    def test_multiple_row_aggregation(self):
        w = pd.Series([[[2, 4, 6],
                        [1, 2, 3],
                        [3, 3, 3]],[
                       [1, 1, 1],
                       [3, 1, 5]]])

        a = Aggregation()
        result = a.transform(w)
        expected_result = pd.Series([[2, 3, 4], [2, 1, 3]])
        pd.testing.assert_series_equal(expected_result, result)

class TestDatasetPreparation(unittest.TestCase):

    def test_series_concatenation(self):
        X = pd.Series([[0.1, 0.2, 0.3], [0.01, 0.1], [0.7, 0.5, 0.9, 0.6]]) # fake news
        Y = pd.Series([[0.9, 0.5], [0.1, 0.2, 0.4, 0.6]]) # real news

        result = pd.concat([X,Y]).reset_index(drop=True)
        expected_result = pd.Series([[0.1, 0.2, 0.3], [0.01, 0.1], [0.7, 0.5, 0.9, 0.6],
                                     [0.9, 0.5], [0.1, 0.2, 0.4, 0.6]])

        pd.testing.assert_series_equal(expected_result, result)

    def test_dataframe_concatenation(self):
        X = pd.DataFrame([[0.1, 0.2, 0.3], [0.01, 0.1], [0.7, 0.5, 0.9, 0.6]])  # fake news
        Y = pd.DataFrame([[0.9, 0.5], [0.1, 0.2, 0.4, 0.6]])  # real news

        result = pd.concat([X, Y]).reset_index(drop=True)
        expected_result = pd.DataFrame([[0.1, 0.2, 0.3], [0.01, 0.1], [0.7, 0.5, 0.9, 0.6],
                                     [0.9, 0.5], [0.1, 0.2, 0.4, 0.6]])

        pd.testing.assert_frame_equal(expected_result, result)

if __name__ == '__main__':
    unittest.main()