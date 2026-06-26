import numpy as np

class ConvolutionalLayer:
    def __init__(self, input_map, filter_maps, stride, activation_function):
        self.input = input_map
        # Sometimes multiple filter maps are used on one input.
        self.filters = filter_maps
        self.stride = stride
        # Current options are ReLu, Leaky ReLu, and Tanh, reasons decribed below.
        self.actfunc = activation_function

    def activation_function(self, x):
        # ReLu is good as it is efficient and mitigates vanishing gradient problems.
        if self.actfunc == "ReLu":
            if x > 0: 
                return x
            else: return 0
        # Leaky ReLu will be helpful in the case where we have dead neurons.
        if self.actfunc == "Leaky ReLu":
            if x > 0:
                return x
            else: return 0.01*x
            # Tanh is good as it is a differentiable function and therefre gradient descent works better.
        if self.actfunc == "Tanh":
            return np.tanh(x)
        
    def forward_propogate(self):
        activation_maps = [self.forward_propogate_one_filter(self.filters[i]) for i in range(len(self.filters))]
        activation_tensor = np.stack(activation_maps)
        return activation_tensor


    def forward_propogate_one_filter(self, filter):
        # Note that it will be important to pick all these dimensions such that the size of the activation map is an integer.
        activation_map = np.zeros(((self.input_map.shape - filter.shape)/self.stride + 1), 
                                  ((self.input_map.shape - filter.shape)/self.stride + 1), 
                                  dtype=float)
        
        for i in range(len(activation_map)):
            for j in range(len(activation_map)):
                # Current input segment here to ensure that we slide over the input map.
                current_input_segment = self.input[self.stride*i:len(filter) + self.stride*i, j:len(filter[0]) + j]
                activation_map[i][j] = self.activation_function(self.linear_model_two_matrices(current_input_segment, filter))
        return activation_map


    def linear_model_two_matrices(self, matrix_input, matrix_weights):
        # np.tensordot doesn't check for this, so this check and method is valuable especially since it also returns a meaningful error message.
        if matrix_input.shape != matrix_weights.shape:
            print("Can't apply linear model to different sized matrices!")
            return

        return np.tensordot(matrix_input, matrix_weights, axes=2)