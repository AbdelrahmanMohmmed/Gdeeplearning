Gdeeplearn
A deep learning package inspired by 'Grokking Deep Learning' by Andrew W. Trask, offering two implementations of the SimpleDL class for neural network computations. The package includes two versions: src/gdeeplearn/simpleDL.py (basic implementation without NumPy) and src/gdeeplearn/simpleDL2.py (enhanced with NumPy and advanced features like GmodelV1).

Installation
This package has two versions with different dependencies:
- simpleDL.py: No external dependencies.
- simpleDL2.py: Requires NumPy for array operations.

To install on another PC, choose one of the following methods:

1. Install from PyPI (after publishing):
   pip install Gdeeplearn
   - For NumPy features, install with extras:
     pip install Gdeeplearn[numpy]

2. Install from source:
   - Clone the repository:
     git clone https://github.com/AbdelrahmanMohmmed/Gdeeplearn.git
   - Navigate to the project directory:
     cd Gdeeplearn
   - Install dependencies using Poetry (recommended):
     poetry install
   - Alternatively, install NumPy manually if using a different environment:
     pip install numpy

Usage
Import the desired SimpleDL class from the respective module and use its methods for neural network calculations.

Example for simpleDL.py (No NumPy)
from gdeeplearn import SimpleDL

# Single input and weight
result1 = SimpleDL.neural_network_one_input(2, 3)  # Returns 6

# Multiple inputs and weights
inputs = [1, 2, 3]
weights = [0.1, 0.2, 0.3]
result2 = SimpleDL.neural_network_multiple_inputs(inputs, weights)  # Returns 1.4

# Multiple outputs
weights = [0.1, 0.2, 0.3]
result3 = SimpleDL.neural_network_multiple_output(2, weights)  # Returns [0.2, 0.4, 0.6]

# Multiple inputs and multiple outputs
inputs = [1, 2, 3]
weight_matrix = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
result4 = SimpleDL.neural_network_multiple_inputs_multiple_outputs(inputs, weight_matrix)  # Returns [1.4, 3.2]

# Hot and cold learning
input_val = 2.0
initial_weight = 0.0
goal_prediction = 4.0
optimized_weight = SimpleDL.hot_cold_learning(input_val, initial_weight, goal_prediction)  # Adjusts weight to ~2.0

Example for simpleDL2.py (With NumPy)
from gdeeplearn import SimpleDL

# Single input and weight (works with arrays)
import numpy as np
input_arr = np.array([1, 2, 3])
weight_arr = np.array([2, 2, 2])
result1 = SimpleDL.neural_network_one_input(input_arr, weight_arr)  # Returns [2, 4, 6]

# Multiple inputs and weights (array support)
inputs = np.array([1, 2, 3])
weights = np.array([0.1, 0.2, 0.3])
result2 = SimpleDL.neural_network_multiple_inputs(inputs, weights)  # Returns 1.4

# Multiple outputs
weights = np.array([0.1, 0.2, 0.3])
result3 = SimpleDL.neural_network_multiple_output(2, weights)  # Returns [0.2, 0.4, 0.6]

# Multiple inputs and multiple outputs
inputs = np.array([1, 2, 3])
weight_matrix = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
result4 = SimpleDL.neural_network_multiple_inputs_multiple_outputs(inputs, weight_matrix)  # Returns [1.4, 3.2]

# Hot and cold learning (array support)
input_val = np.array([2.0, 3.0])
initial_weight = np.array([0.0, 0.0])
goal_prediction = np.array([4.0, 6.0])
optimized_weight = SimpleDL.hot_cold_learning(input_val, initial_weight, goal_prediction)  # Adjusts weights to ~[2.0, 2.0]

# Gradient descent learning
input_val = np.array([2.0, 3.0])
initial_weight = np.array([0.0, 0.0])
goal_pred = np.array([4.0, 6.0])
optimized_weight_gd = SimpleDL.gradient_descent_learning(input_val, initial_weight, goal_pred)  # Adjusts weights toward ~[2.0, 2.0]

# Train with GmodelV1
n_samples, n_features = 5, 3
input_data = np.random.random((n_samples, n_features))
labels = np.zeros((n_samples, 10))
labels[np.arange(n_samples), np.random.randint(0, 10, n_samples)] = 1
model = SimpleDL(alpha=0.01, epochs=2, hidden_size=4, num_labels=10)
error, accuracy = model.GmodelV1(input_data, labels)  # Trains model and returns error, accuracy

Testing
Run the included tests with Poetry for both versions:
poetry run python -m unittest tests/test_simpledl.py  # Tests simpleDL.py
poetry run python -m unittest tests/test_simpledl2.py  # Tests simpleDL2.py
Note: Ensure NumPy is installed for simpleDL2.py tests.

License
MIT License (add a LICENSE file with your preferred terms).

Contributing
Feel free to submit issues or pull requests on GitHub!

Last Updated
Last updated at 10:06 PM EEST on Friday, August 15, 2025.