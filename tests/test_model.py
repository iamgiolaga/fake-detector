import unittest

from mulearn import FuzzyInductor


class TestModel(unittest.TestCase):

    def test_adjustment(self):
        exception = "Objective Q not PSD (diagonal adjustment of 1.1e+02 would be required). Set NonConvex parameter to 2 to solve model."
        result = exception.split("adjustment of ")
        result = result[1].split(" would be")
        assert float(result[0]) == 1.1e+02

    def test_fuzzifier(self):
        f = FuzzyInductor()
        print(f.get_params())