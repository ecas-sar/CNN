import numpy as np

class ConvolutionalLayer:
    def __init__(self, input_map, num_filters, filter_size, stride, activation_function):
        self.input_map = input_map
        # Initial weights are multiplied by 0.01 in order to ensure stability during training.
        self.filters = [np.random.randn(filter_size, filter_size) * 0.01 for _ in range(num_filters)]
        self.stride = stride
        # Current options are ReLu, Leaky ReLu, and Tanh, reasons decribed below in activation_function.
        self.actfunc = activation_function

    """Note that the activation function is being used here rather than 
    in an activation layer class because there is little point having a whole other file just for the sake of applying
    an activation function to the activation map when it can easily be done here, this makes the codebase look cleaner."""
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
        activation_maps = [self.forward_propogate_one_filter(self.filters[i]) for i in range(len(self.filters))]
        activation_tensor = np.stack(activation_maps)
        return activation_tensor


    def forward_propogate_one_filter(self, filter):
        # Note that it will be important to pick all these dimensions such that the size of the activation map is an integer.
        # Note that shape returns a tuple but input_map and filter are assumed to be square, so we can just get the number at the 0th index.
        activation_map = np.zeros((((self.input_map.shape[0] - filter.shape[0])/self.stride + 1), 
                                  ((self.input_map.shape[0] - filter.shape[0])/self.stride + 1)), 
                                  dtype=float)
        
        for i in range(len(activation_map)):
            for j in range(len(activation_map[0])):
                # Current input segment here to ensure that we slide over the input map.
                current_input_segment = self.input_map[self.stride*i:len(filter) + self.stride*i, self.stride*j:len(filter[0]) + self.stride*j]
                # We apply activation here also because it would be needlessly expensive to iterate over the whole tensory AGAIN just to apply activation to each element.
                activation_map[i][j] = self.activation_function(self.linear_model_two_matrices(current_input_segment, filter))
        return activation_map


    def linear_model_two_matrices(self, matrix_input, matrix_weights):
        # np.tensordot doesn't check for this, so this check and method is valuable especially since it also returns a meaningful error message.
        if matrix_input.shape != matrix_weights.shape:
            print("Can't apply linear model to different sized matrices!")
            return

        return np.tensordot(matrix_input, matrix_weights, axes=2)