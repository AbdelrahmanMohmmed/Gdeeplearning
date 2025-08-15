import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
from gdeeplearn.simpleDL2 import SimpleDL

class TestSimpleDL2(unittest.TestCase):
    def setUp(self):
        """Set up a SimpleDL instance for tests requiring an instance."""
        self.model = SimpleDL(alpha=0.01, epochs=2, hidden_size=4, num_labels=10)

    def test_neural_network_one_input(self):
        # Test with scalars
        self.assertEqual(SimpleDL.neural_network_one_input(2, 3), 6)
        self.assertEqual(SimpleDL.neural_network_one_input(0, 5), 0)
        with self.assertRaises(ValueError):
            SimpleDL.neural_network_one_input("2", 3)  # Should raise ValueError for string
        # Test with arrays
        input_arr = np.array([1, 2, 3])
        weight_arr = np.array([2, 2, 2])
        np.testing.assert_array_equal(SimpleDL.neural_network_one_input(input_arr, weight_arr), np.array([2, 4, 6]))
        # Test array-scalar multiplication
        np.testing.assert_array_equal(SimpleDL.neural_network_one_input(input_arr, 2), np.array([2, 4, 6]))

    def test_neural_network_multiple_inputs(self):
        # Test with lists
        inputs = [1, 2, 3]
        weights = [0.1, 0.2, 0.3]
        self.assertAlmostEqual(SimpleDL.neural_network_multiple_inputs(inputs, weights), 1.4, places=7)
        with self.assertRaises(ValueError):
            SimpleDL.neural_network_multiple_inputs([1, 2], [1, 2, 3])
        with self.assertRaises(ValueError):
            SimpleDL.neural_network_multiple_inputs([], [])
        with self.assertRaises(ValueError):
            SimpleDL.neural_network_multiple_inputs([1, "a"], [0.1, 0.2])
        # Test with arrays
        input_arr = np.array([1, 2, 3])
        weight_arr = np.array([0.1, 0.2, 0.3])
        self.assertAlmostEqual(SimpleDL.neural_network_multiple_inputs(input_arr, weight_arr), 1.4, places=7)
        # Test mismatched shapes
        with self.assertRaises(ValueError):
            SimpleDL.neural_network_multiple_inputs(np.array([1, 2]), np.array([0.1, 0.2, 0.3]))

    def test_neural_network_multiple_output(self):
        # Test with scalar and list
        input_val = 2
        weights = [0.1, 0.2, 0.3]
        result = SimpleDL.neural_network_multiple_output(input_val, weights)
        self.assertEqual(result.tolist(), [0.2, 0.4, 0.6])
        with self.assertRaises(ValueError):
            SimpleDL.neural_network_multiple_output(2, [])
        # Test with arrays
        input_arr = np.array([1, 2])
        weight_arr = np.array([0.1, 0.2])
        np.testing.assert_array_equal(SimpleDL.neural_network_multiple_output(input_arr, weight_arr), np.array([0.1, 0.4]))
        # Test array-scalar
        np.testing.assert_array_equal(SimpleDL.neural_network_multiple_output(2, weight_arr), np.array([0.2, 0.4]))

    def test_neural_network_multiple_inputs_multiple_outputs(self):
        # Test with lists
        inputs = [1, 2, 3]
        weights = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        result = SimpleDL.neural_network_multiple_inputs_multiple_outputs(inputs, weights)
        self.assertAlmostEqual(result[0], 1.4, places=7)
        self.assertAlmostEqual(result[1], 3.2, places=7)
        with self.assertRaises(ValueError):
            SimpleDL.neural_network_multiple_inputs_multiple_outputs([1, 2], [[0.1, 0.2, 0.3]])
        # Test with arrays
        input_arr = np.array([1, 2, 3])
        weight_arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        result_arr = SimpleDL.neural_network_multiple_inputs_multiple_outputs(input_arr, weight_arr)
        np.testing.assert_array_almost_equal(result_arr, np.array([1.4, 3.2]), decimal=7)
        # Test empty input
        with self.assertRaises(ValueError):
            SimpleDL.neural_network_multiple_inputs_multiple_outputs([], weight_arr)

    def test_hot_cold_learning(self):
        # Test with scalars
        input_val = 2.0
        initial_weight = 0.0
        goal_prediction = 4.0
        optimized_weight = SimpleDL.hot_cold_learning(input_val, initial_weight, goal_prediction)
        self.assertAlmostEqual(optimized_weight, 2.0, delta=0.001)
        # Test with small step_size and increased epochs
        optimized_weight_small_step = SimpleDL.hot_cold_learning(input_val, initial_weight, goal_prediction, step_size=0.0001, epochs=10000)
        self.assertAlmostEqual(optimized_weight_small_step, 2.0, delta=0.001)
        # Test with arrays
        input_arr = np.array([2.0, 3.0])
        weight_arr = np.array([0.0, 0.0])
        goal_arr = np.array([4.0, 6.0])
        result_arr = SimpleDL.hot_cold_learning(input_arr, weight_arr, goal_arr)
        np.testing.assert_array_almost_equal(result_arr, np.array([2.0, 2.0]), decimal=1)

    def test_gradient_descent_learning(self):
        # Test with scalars
        input_val = 2.0
        initial_weight = 0.0
        goal_pred = 4.0
        optimized_weight = SimpleDL.gradient_descent_learning(input_val, initial_weight, goal_pred)
        self.assertAlmostEqual(optimized_weight, 2.0, delta=0.01)
        optimized_weight_small_alpha = SimpleDL.gradient_descent_learning(input_val, initial_weight, goal_pred, alpha=0.001)
        self.assertAlmostEqual(optimized_weight_small_alpha, 2.0, delta=0.05)
        # Test with arrays
        input_arr = np.array([2.0, 3.0])
        weight_arr = np.array([0.0, 0.0])
        goal_arr = np.array([4.0, 6.0])
        result_arr = SimpleDL.gradient_descent_learning(input_arr, weight_arr, goal_arr)
        np.testing.assert_array_almost_equal(result_arr, np.array([2.0, 2.0]), decimal=1)

    def test_relu(self):
        # Test with scalar
        self.assertEqual(self.model.relu(1.0), 1.0)
        self.assertEqual(self.model.relu(-1.0), 0.0)
        # Test with array
        input_arr = np.array([-1.0, 0.0, 1.0])
        result_arr = self.model.relu(input_arr)
        np.testing.assert_array_equal(result_arr, np.array([0.0, 0.0, 1.0]))

    def test_relu_derivative(self):
        # Test with scalar
        self.assertEqual(self.model.relu_derivative(1.0), 1.0)
        self.assertEqual(self.model.relu_derivative(-1.0), 0.0)
        # Test with array
        input_arr = np.array([-1.0, 0.0, 1.0])
        result_arr = self.model.relu_derivative(input_arr)
        np.testing.assert_array_equal(result_arr, np.array([0.0, 0.0, 1.0]))

    def test_GmodelV1(self):
        # Create dummy data
        n_samples, n_features = 5, 3
        input_data = np.random.random((n_samples, n_features))
        labels = np.zeros((n_samples, 10))
        labels[np.arange(n_samples), np.random.randint(0, 10, n_samples)] = 1
        # Use the instance from setUp
        error, accuracy = self.model.GmodelV1(input_data, labels)
        self.assertIsInstance(error, float)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(error, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        # Test with different hyperparameters
        model2 = SimpleDL(alpha=0.1, epochs=1, hidden_size=5, num_labels=5)
        error2, accuracy2 = model2.GmodelV1(input_data[:, :2], labels[:, :5])  # Adjusted shapes
        self.assertIsInstance(error2, float)
        self.assertIsInstance(accuracy2, float)

if __name__ == '__main__':
    unittest.main()