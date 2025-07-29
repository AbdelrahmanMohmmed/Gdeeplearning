class SimpleDL:
    """A simple deep learning class inspired by basic neural network concepts."""

    def __init__(self):
        """Initialize the SimpleDL class."""
        pass

    @staticmethod
    def neural_network_one_input(input_val, weight):
        """
        Compute the prediction for a single input and weight.
        
        Args:
            input_val (float): The input value.
            weight (float): The weight value.
        
        Returns:
            float: The prediction (input * weight).
        """
        prediction = input_val * weight
        return prediction

    @staticmethod
    def neural_network_multiple_inputs(input_list, weight_list):
        """
        Compute the weighted sum for multiple inputs and weights.
        
        Args:
            input_list (list): List of input values.
            weight_list (list): List of corresponding weights.
        
        Returns:
            float: The sum of (input * weight) for all pairs.
        
        Raises:
            AssertionError: If the lengths of input_list and weight_list differ.
        """
        assert len(input_list) == len(weight_list)
        output = 0
        for i in range(len(input_list)):
            output += (input_list[i] * weight_list[i])
        return output

    @staticmethod
    def neural_network_multiple_output(input_val, weight_list):
        """
        Compute multiple outputs using a single input and multiple weights.
        
        Args:
            input_val (float): The single input value.
            weight_list (list): List of weight values.
        
        Returns:
            list: List of predictions (input * weight for each weight).
        """
        output = [0] * len(weight_list)
        for i in range(len(weight_list)):
            output[i] = input_val * weight_list[i]
        return output

    @staticmethod
    def neural_network_multiple_inputs_multiple_outputs(input_list, weight_matrix):
        """
        Compute multiple outputs for multiple inputs using a weight matrix.
        
        Args:
            input_list (list): List of input values.
            weight_matrix (list): List of weight lists, where each inner list
                                corresponds to weights for an output.
        
        Returns:
            list: List of outputs, each computed as a weighted sum of inputs.
        
        Raises:
            AssertionError: If the number of inputs doesn't match the length of
                           each weight list, or if weight_matrix is empty.
        """
        assert len(input_list) == len(weight_matrix[0])
        output = [0] * len(weight_matrix)
        for i in range(len(weight_matrix)):
            output[i] = SimpleDL.neural_network_multiple_inputs(input_list, weight_matrix[i])
        return output
    
    @staticmethod
    def hot_cold_learning(input_val, weight, goal_prediction, step_size=0.01, epochs=1101, tolerance=0.00001):
        """
        Implements a hot and cold learning algorithm to adjust weight toward a goal prediction.
        
        Args:
            input_val (float): The input value to the neuron.
            weight (float): The initial weight to be adjusted.
            goal_prediction (float): The target output value.
            step_size (float, optional): The amount to adjust the weight. Defaults to 0.01.
            epochs (int, optional): The maximum number of iterations. Defaults to 1101.
            tolerance (float, optional): The error threshold to stop learning. Defaults to 0.00001.
        
        Returns:
            float: The optimized weight after learning.
        
        Notes:
            Adjusts the weight by testing small increases and decreases, moving in the direction
            that reduces the squared error. Stops if the error falls below the tolerance or
            reaches the maximum epochs.
        """
        current_weight = weight
        for iteration in range(epochs):
            prediction = input_val * current_weight
            error = (prediction - goal_prediction) ** 2
            print(f"Iteration {iteration}, Error: {error:.6f}, Prediction: {prediction:.6f}")
            
            up_prediction = input_val * (current_weight + step_size)
            up_error = (up_prediction - goal_prediction) ** 2
            down_prediction = input_val * (current_weight - step_size)
            down_error = (down_prediction - goal_prediction) ** 2
            
            if error < tolerance:
                break
                
            if down_error < up_error:
                current_weight -= step_size
            elif up_error < down_error:
                current_weight += step_size
                
        return current_weight

