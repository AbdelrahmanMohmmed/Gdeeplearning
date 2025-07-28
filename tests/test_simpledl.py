import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
from gdeeplearn.simpleDL import SimpleDL

class TestSimpleDL(unittest.TestCase):
    def test_neural_network_one_input(self):
        self.assertEqual(SimpleDL.neural_network_one_input(2, 3), 6)
        self.assertEqual(SimpleDL.neural_network_one_input(0, 5), 0)

    def test_neural_network_multiple_inputs(self):
        inputs = [1, 2, 3]
        weights = [0.1, 0.2, 0.3]
        self.assertAlmostEqual(SimpleDL.neural_network_multiple_inputs(inputs, weights), 1.4, places=7)
        with self.assertRaises(AssertionError):
            SimpleDL.neural_network_multiple_inputs([1, 2], [1])

    def test_neural_network_multiple_output(self):
        input_val = 2
        weights = [0.1, 0.2, 0.3]
        result = SimpleDL.neural_network_multiple_output(input_val, weights)
        self.assertEqual(result, [0.2, 0.4, 0.6])  # Exact matches here are fine

    def test_neural_network_multiple_inputs_multiple_outputs(self):
        inputs = [1, 2, 3]
        weights = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        result = SimpleDL.neural_network_multiple_inputs_multiple_outputs(inputs, weights)
        self.assertAlmostEqual(result[0], 1.4, places=7)  # Check each element
        self.assertAlmostEqual(result[1], 3.2, places=7)
        with self.assertRaises(AssertionError):
            SimpleDL.neural_network_multiple_inputs_multiple_outputs([1, 2], [[0.1, 0.2, 0.3]])

if __name__ == '__main__':
    unittest.main()