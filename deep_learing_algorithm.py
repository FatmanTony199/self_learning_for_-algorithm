layer_dims = [X_train.shape[0], 4, 2, 1]
larning_rate = 0.0045
num_iterations = 5000

def initialize_parameters(layer_dims):
	
    np.random.seed(1)
    parameters = {}
    layers = len(layer_dims) 
    
    for layer in range(1,layers):
        parameters["W"+str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer-1])
        parameters["b"+str(layer)] = np.zeros((layer_dims[layer], 1))
        
        assert(parameters['W' + str(layer)].shape == (layer_dims[layer], layer_dims[layer-1]))
        assert(parameters['b' + str(layer)].shape == (layer_dims[layer], 1))
                                              
    return parameters