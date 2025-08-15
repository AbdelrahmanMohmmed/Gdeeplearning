import numpy as np
from typing import Union, Tuple, List

class SimpleDL:
    """A professional deep learning class inspired by basic neural network concepts from
    'Grokking Deep Learning' by Andrew W. Trask.
    
    This class provides instance and static methods for neural network computations and training.
    """

    # Class-level defaults (can be overridden in __init__)
    DEFAULT_ALPHA = 0.01
    DEFAULT_EPOCHS = 1000
    DEFAULT_TOLERANCE = 0.00001
    DEFAULT_HIDDEN_SIZE = 10
    DEFAULT_NUM_LABELS = 10

    def __init__(self, alpha: float = DEFAULT_ALPHA, epochs: int = DEFAULT_EPOCHS,
                 tolerance: float = DEFAULT_TOLERANCE, hidden_size: int = DEFAULT_HIDDEN_SIZE,
                 num_labels: int = DEFAULT_NUM_LABELS):
        """Initialize the SimpleDL class with configurable hyperparameters.
        
        Args:
            alpha (float, optional): Learning rate for gradient descent. Defaults to 0.01.
            epochs (int, optional): Maximum number of training iterations. Defaults to 1000.
            tolerance (float, optional): Error threshold for early stopping. Defaults to 0.00001.
            hidden_size (int, optional): Number of neurons in the hidden layer. Defaults to 10.
            num_labels (int, optional): Number of output labels. Defaults to 10.
        """
        self.alpha = alpha
        self.epochs = epochs
        self.tolerance = tolerance
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        # Initialize weights as None; they will be set in GmodelV1
        self.weights_0_1 = None
        self.weights_1_2 = None

    @staticmethod
    def neural_network_one_input(input_val: Union[float, np.ndarray], weight: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the prediction for a single input and weight.
        
        Args:
            input_val (Union[float, np.ndarray]): The input value or array.
            weight (Union[float, np.ndarray]): The weight value or array.
        
        Returns:
            Union[float, np.ndarray]: The prediction (input * weight).
        
        Raises:
            ValueError: If input_val or weight is not numeric or compatible for multiplication.
        """
        if not np.isscalar(input_val) and not isinstance(input_val, np.ndarray):
            raise ValueError("Input must be a scalar or NumPy array.")
        if not np.isscalar(weight) and not isinstance(weight, np.ndarray):
            raise ValueError("Weight must be a scalar or NumPy array.")
        if isinstance(input_val, str) or isinstance(weight, str):
            raise ValueError("Input and weight must be numeric, not strings.")
        return np.multiply(input_val, weight)

    @staticmethod
    def neural_network_multiple_inputs(input_list: Union[List[float], np.ndarray], weight_list: Union[List[float], np.ndarray]) -> float:
        """Compute the weighted sum for multiple inputs and weights.
        
        Args:
            input_list (Union[List[float], np.ndarray]): List or array of input values.
            weight_list (Union[List[float], np.ndarray]): List or array of corresponding weights.
        
        Returns:
            float: The sum of (input * weight) for all pairs.
        
        Raises:
            ValueError: If lists/arrays have different lengths or contain non-numeric values.
            ValueError: If lists/arrays are empty.
        """
        input_array = np.array(input_list) if not isinstance(input_list, np.ndarray) else input_list
        weight_array = np.array(weight_list) if not isinstance(weight_list, np.ndarray) else weight_list
        if input_array.size == 0 or weight_array.size == 0:
            raise ValueError("Input and weight arrays cannot be empty.")
        if input_array.shape != weight_array.shape:
            raise ValueError("Input and weight arrays must have the same shape.")
        if not np.issubdtype(input_array.dtype, np.number) or not np.issubdtype(weight_array.dtype, np.number):
            raise ValueError("All values must be numeric.")
        return np.sum(input_array * weight_array)

    @staticmethod
    def neural_network_multiple_output(input_val: Union[float, np.ndarray], weight_list: Union[List[float], np.ndarray]) -> np.ndarray:
        """Compute multiple outputs using a single input and multiple weights.
        
        Args:
            input_val (Union[float, np.ndarray]): The single input value or array.
            weight_list (Union[List[float], np.ndarray]): Array of weight values.
        
        Returns:
            np.ndarray: Array of predictions (input * weight for each weight).
        
        Raises:
            ValueError: If weight_list is empty or contains non-numeric values.
        """
        weight_array = np.array(weight_list) if not isinstance(weight_list, np.ndarray) else weight_list
        if weight_array.size == 0:
            raise ValueError("Weight array cannot be empty.")
        if not np.issubdtype(weight_array.dtype, np.number):
            raise ValueError("All weights must be numeric.")
        return np.multiply(input_val, weight_array)

    @staticmethod
    def neural_network_multiple_inputs_multiple_outputs(input_list: Union[List[float], np.ndarray], weight_matrix: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """Compute multiple outputs for multiple inputs using a weight matrix.
        
        Args:
            input_list (Union[List[float], np.ndarray]): List or array of input values.
            weight_matrix (Union[List[List[float]], np.ndarray]): Array of weight arrays, where each row
                                                               corresponds to weights for an output.
        
        Returns:
            np.ndarray: Array of outputs, each computed as a weighted sum of inputs.
        
        Raises:
            ValueError: If input_list or weight_matrix is empty, or if dimensions mismatch.
            ValueError: If any value is non-numeric.
        """
        input_array = np.array(input_list) if not isinstance(input_list, np.ndarray) else input_list
        weight_array = np.array(weight_matrix) if not isinstance(weight_matrix, np.ndarray) else weight_matrix
        if input_array.size == 0 or weight_array.size == 0:
            raise ValueError("Input array and weight matrix cannot be empty.")
        if weight_array.shape[1] != input_array.shape[0]:
            raise ValueError("Weight matrix columns must match input array length.")
        if not np.issubdtype(input_array.dtype, np.number) or not np.issubdtype(weight_array.dtype, np.number):
            raise ValueError("All values must be numeric.")
        return np.dot(weight_array, input_array)

    @staticmethod
    def hot_cold_learning(input_val: Union[float, np.ndarray], weight: Union[float, np.ndarray], goal_prediction: Union[float, np.ndarray],
                         step_size: float = 0.01, epochs: int = 1101, tolerance: float = 0.00001) -> Union[float, np.ndarray]:
        """Implements a hot and cold learning algorithm to adjust weight toward a goal prediction.
        
        Args:
            input_val (Union[float, np.ndarray]): The input value or array to the neuron.
            weight (Union[float, np.ndarray]): The initial weight or array to be adjusted.
            goal_prediction (Union[float, np.ndarray]): The target output value or array.
            step_size (float, optional): The amount to adjust the weight. Defaults to 0.01.
            epochs (int, optional): The maximum number of iterations. Defaults to 1101.
            tolerance (float, optional): The error threshold to stop learning. Defaults to 0.00001.
        
        Returns:
            Union[float, np.ndarray]: The optimized weight or array after learning.
        
        Notes:
            Adjusts the weight by testing small increases and decreases, moving in the direction
            that reduces the squared error. Stops if the error falls below the tolerance or
            reaches the maximum epochs.
        """
        current_weight = np.array(weight) if not isinstance(weight, np.ndarray) else weight
        for iteration in range(epochs):
            prediction = SimpleDL.neural_network_one_input(input_val, current_weight)
            error = np.square(prediction - goal_prediction)
            # Handle array printing
            if np.size(error) > 1:
                print(f"Iteration {iteration}, Error: {np.array2string(error, precision=6)}, Prediction: {np.array2string(prediction, precision=6)}")
            else:
                print(f"Iteration {iteration}, Error: {error:.6f}, Prediction: {prediction:.6f}")
            
            up_prediction = SimpleDL.neural_network_one_input(input_val, current_weight + step_size)
            up_error = np.square(up_prediction - goal_prediction)
            down_prediction = SimpleDL.neural_network_one_input(input_val, current_weight - step_size)
            down_error = np.square(down_prediction - goal_prediction)
            
            if np.all(error < tolerance):
                break
                
            if np.all(down_error < up_error):
                current_weight -= step_size
            elif np.all(up_error < down_error):
                current_weight += step_size
                
        return current_weight

    @staticmethod
    def gradient_descent_learning(input_val: Union[float, np.ndarray], weight: Union[float, np.ndarray], goal_pred: Union[float, np.ndarray],
                                 alpha: float = 0.01, epochs: int = 1000, tolerance: float = 0.00001) -> Union[float, np.ndarray]:
        """Adjusts the weight using iterative gradient descent based on the prediction error.
        
        Args:
            input_val (Union[float, np.ndarray]): The input value or array to the neuron.
            weight (Union[float, np.ndarray]): The initial weight or array to be adjusted.
            goal_pred (Union[float, np.ndarray]): The target prediction value or array.
            alpha (float, optional): The learning rate. Defaults to 0.01.
            epochs (int, optional): The maximum number of iterations. Defaults to 1000.
            tolerance (float, optional): The error threshold to stop learning. Defaults to 0.00001.
        
        Returns:
            Union[float, np.ndarray]: The optimized weight or array after iterative gradient descent.
        
        Notes:
            Iteratively updates the weight using the gradient descent rule: weight -= input * delta * alpha,
            where delta is the difference between prediction and goal. Stops if the error falls below
            the tolerance or reaches the maximum epochs.
        """
        current_weight = np.array(weight) if not isinstance(weight, np.ndarray) else weight
        for iteration in range(epochs):
            pred = SimpleDL.neural_network_one_input(input_val, current_weight)
            error = np.square(pred - goal_pred)
            delta = pred - goal_pred
            weight_delta = np.multiply(input_val, delta)
            current_weight -= np.multiply(weight_delta, alpha)
            # Handle array printing
            if np.size(error) > 1:
                print(f"Iteration {iteration}, Error: {np.array2string(error, precision=6)}, Prediction: {np.array2string(pred, precision=6)}, Weight: {np.array2string(current_weight, precision=6)}")
            else:
                print(f"Iteration {iteration}, Error: {error:.6f}, Prediction: {pred:.6f}, Weight: {current_weight:.6f}")

            if np.all(error < tolerance):
                break
                
        return current_weight

    def relu(self, input_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Applies the ReLU activation function to the input value or array.
        
        Args:
            input_val (Union[float, np.ndarray]): The input value or array to apply ReLU on.
        
        Returns:
            Union[float, np.ndarray]: The output after applying ReLU (max(0, input_val)).
        
        Notes:
            Uses NumPy's vectorized maximum for efficient array operations.
            Handles edge cases like inf and nan gracefully.
        """
        input_array = np.array(input_val) if not isinstance(input_val, np.ndarray) else input_val
        return np.maximum(0, input_array)

    def relu_derivative(self, input_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Computes the derivative of the ReLU activation function for the input value or array.
        
        Args:
            input_val (Union[float, np.ndarray]): The input value or array to compute the derivative for.
        
        Returns:
            Union[float, np.ndarray]: The derivative of ReLU (1 if input_val > 0, else 0).
        
        Notes:
            Uses NumPy's vectorized comparison for efficient array operations.
            Returns 0 for nan or inf inputs to maintain stability.
        """
        input_array = np.array(input_val) if not isinstance(input_val, np.ndarray) else input_val
        return np.where(input_array > 0, 1.0, 0.0)

    def GmodelV1(self, input_data: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """Trains a simple neural network model with one hidden layer using gradient descent.
        
        Args:
            input_data (np.ndarray): Input data with shape (n_samples, n_features).
            labels (np.ndarray): Target labels with shape (n_samples, n_labels).
        
        Returns:
            Tuple[float, float]: Average error and accuracy over the dataset.
        
        Notes:
            Initializes weights randomly and updates them using backpropagation with ReLU activation.
            Progress is printed to stdout during training.
        """
        if self.weights_0_1 is None or self.weights_1_2 is None:
            self.weights_0_1 = 0.2 * np.random.random((input_data.shape[1], self.hidden_size)) - 0.1
            self.weights_1_2 = 0.2 * np.random.random((self.hidden_size, self.num_labels)) - 0.1

        total_error, correct_count = 0.0, 0
        for j in range(self.epochs):
            error, correct_cnt = 0.0, 0
            for i in range(len(input_data)):
                layer_0 = input_data[i:i+1]
                layer_1 = self.relu(np.dot(layer_0, self.weights_0_1))
                layer_2 = np.dot(layer_1, self.weights_1_2)
                error += np.sum(np.square(labels[i:i+1] - layer_2))
                correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))

                layer_2_delta = labels[i:i+1] - layer_2
                layer_1_delta = np.dot(layer_2_delta, self.weights_1_2.T) * self.relu_derivative(layer_1)
                self.weights_1_2 += self.alpha * np.dot(layer_1.T, layer_2_delta)
                self.weights_0_1 += self.alpha * np.dot(layer_0.T, layer_1_delta)

            total_error += error / len(input_data)
            correct_count += correct_cnt / len(input_data)
            print(f"\rI: {j}, Error: {error/float(len(input_data)):.5f}, "
                  f"Correct: {correct_cnt/float(len(input_data)):.5f}", end='')

        avg_error = total_error / self.epochs
        avg_accuracy = correct_count / self.epochs
        return avg_error, avg_accuracy