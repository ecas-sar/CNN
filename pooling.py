import numpy as np
class PoolingLayer:
    def __init__(self, input_map, filter_size, stride, pool_method):
        self.input_map = input_map
        self.filter_size = filter_size
        self.stride = stride
        # Current options for pooling methods are given in pool_one_segment, with reasons given for using them.
        self.pool_method = pool_method

    def pool_everything(self):
        pooled_maps = [self.pool_each_layer(self.input_map[i]) for i in range(len(self.input_map[0][0]))]
        pooled_tensor = np.stack(pooled_maps)
        return pooled_tensor

    def pool_each_layer(self, input_layer):
        pooled_map = np.zeros((((input_layer.shape[0] - self.filter_size)/self.stride + 1), 
                               ((input_layer.shape[1] - self.filter_size)/self.stride + 1)),
                               dtype=float)
        
        # Same logic as in convolutional layer, but pools everything rather than applying linear model.
        for i in range(len(pooled_map)):
            for j in range(len(pooled_map[0])):
                current_input_segment = input_layer[self.stride*i:self.filter_size + self.stride*i, self.stride*j:self.filter_size + self.stride*j]
                pooled_map[i][j] = self.pool_one_segment(current_input_segment)
        
        return pooled_map

    def pool_one_segment(self, matrix_to_pool):
        # Retains important features like edges and textures, focuses on strongest features.
        if self.pool_method == "max":
            return matrix_to_pool.max()
        # Represents overall features (which can be more useful than strongest in some cases), and is also smoother than max.
        elif self.pool_method == "average":
            return matrix_to_pool.mean()