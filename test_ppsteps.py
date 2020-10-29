import unittest

from classes.ppsteps import DuplicateRemoval

class TestDuplicateRemoval(unittest.TestCase):
    def test_input(self):
        self.assertEqual(2,2)

    def test_values(self):
        self.assertRaises(ValueError, DuplicateRemoval.transform, 2)

if __name__ == '__main__':
    unittest.main()