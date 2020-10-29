import unittest
import pandas as pd
import pandas.testing as pdt

from ppsteps import DuplicateRowsRemoval, BadCharRemoval, DuplicateWordsRemoval, Lowercasing, Lemmatization, \
    NumbersRemoval, RemoveWordsWithNumbers, CleaningWords, Stemming, StopwordRemoval


## DESCRIPTION ##
# This file defines the unit testing

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

if __name__ == '__main__':
    unittest.main()