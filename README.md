Gdeeplearn
A simple deep learning package inspired by the concepts in Grokking Deep Learning by Andrew W. Trask. The package is located in the src/gdeeplearn directory.
Installation
This package has no external dependencies. To use it, clone the repository and install it using Poetry:
git clone https://github.com/yourusername/Gdeeplearn.git
cd Gdeeplearn
poetry install

Usage
Import the SimpleDL class from the gdeeplearn package and use its static methods for neural network calculations.
Example
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

Testing
Run the included tests with Poetry:
poetry run python -m unittest tests/test_simpledl.py

License
MIT License (add a LICENSE file with your preferred terms).
Contributing
Feel free to submit issues or pull requests on GitHub!