import numpy as np
class FullyConnected:
    def __init__(self, feature_map, output_size, activation_function=None,):
        self.feature_map = feature_map
        self.feature_vector = feature_map.flatten()
        # Number of neurons.
        self.output_size = output_size
        # This activation function is here in case there are multiple FC layers, the proper classification scores will be done in the output layer.
        self.actfunc = activation_function

        # The weights are multiplied by 0.01 to keep initial guess for weights small, this helps stability during training.
        self.weights_matrix = np.random.randn(output_size, len(self.feature_vector)) * 0.01
        self.bias_vector = np.zeros(output_size)

    def activation_function(self, x):
        # ReLu is good as it is efficient and mitigates vanishing gradient problems.
        if self.actfunc == "ReLu":
            if x > 0: 
                return x
            else: return 0
        # Leaky ReLu will be helpful in the case where we have dead neurons.
        elif self.actfunc == "Leaky ReLu":
            if x > 0:
                return x
            else: return 0.01*x
        # Tanh is good as it is a differentiable function and therefore gradient descent works better.
        elif self.actfunc == "Tanh":
            return np.tanh(x)
        else: return x

    def forward_propogate(self):
        output = self.weights_matrix @ self.feature_vector + self.bias_vector
        # Apply activation element wise to the vector output.
        return np.vectorize(self.activation_function)(output)
        
    