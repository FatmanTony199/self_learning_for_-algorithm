layer_dims = [X_train.shape[0], 4, 2, 1]
larning_rate = 0.0045
num_iterations = 5000

def initialize_parameters(layer_dims):

	np.random.seed(1)
	parameters = {}
	ayers = len(layer_dims) 
    
	for layer in range(1,layers):
		parameters["W"+str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer-1])
		parameters["b"+str(layer)] = np.zeros((layer_dims[layer], 1))
        
		assert(parameters['W' + str(layer)].shape == (layer_dims[layer], layer_dims[layer-1]))
		assert(parameters['b' + str(layer)].shape == (layer_dims[layer], 1))
	                                              
	return parameters

def linear_forward(A, W, b):

	Z = np.dot(W,A)+b
	cache = (A,W,b)
	assert(Z.shape == (W.shape[0], A.shape[1]))
	return Z, cache
    
def linear_activation_forward(A_pre, W, b, activation):

	if activation == "sigmoid":
		Z, linear_cache = linear_forward(A_pre, W, b)
		A, activation_cache = sigmoid(Z)
	elif activation == "relu":
		Z, linear_cache = linear_forward(A_pre, W, b)
		A, activation_cache = relu(Z)

	cache = (linear_cache, activation_cache)
	assert (A.shape == (W.shape[0], A_pre.shape[1]))
	return A, cache