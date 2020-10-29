import unittest
import pandas as pd
import pandas.testing as pdt

from ppsteps import DuplicateRowsRemoval, BadCharRemoval, DuplicateWordsRemoval, Lowercasing


## DESCRIPTION ##
# This file defines the unit testing

class TestLowercasing(unittest.TestCase):
    # input should be a string

    def test_single_row_output(self):
        example = pd.Series(["Donald DoNald tRump"])
        l = Lowercasing()
        result = l.transform(example)
        expected_result = pd.Series(["donald donald trump"])
        pd.testing.assert_series_equal(expected_result, result)

    def test_multi_row_output(self):
        example = pd.Series(["Donald DoNald tRump","thiS iS AnOther wrong STRING"])
        l = Lowercasing()
        result = l.transform(example)
        expected_result = pd.Series(["donald donald trump", "this is another wrong string"])
        pd.testing.assert_series_equal(expected_result, result)

class TestDuplicateWordsRemoval(unittest.TestCase):
    # input should be a list of strings

    def test_single_row_output(self):
        example = pd.Series([["donald", "donald", "trump"]])
        d = DuplicateWordsRemoval()
        result = d.transform(example)
        expected_result = pd.Series([["donald", "trump"]])
        pd.testing.assert_series_equal(expected_result, result)

    def test_multi_row_output(self):
        example = pd.Series([["donald", "donald", "trump"], ["this", "is", "is", "a", "duplicate", "word", "word"]])
        d = DuplicateWordsRemoval()
        result = d.transform(example)
        expected_result = pd.Series([["donald", "trump"], ["this", "is", "a", "duplicate", "word"]])
        pd.testing.assert_series_equal(expected_result, result)
if __name__ == '__main__':
    unittest.main()