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

def sigmoid(Z):
	A = 1 / (1+np.exp(-Z))
	cache = Z
	return A, cache

def relu(Z):
	A = np.maximum(0,Z)
	assert(A.shape == Z.shape)
	cache = Z
	return A, cache

def model_forward(X, parameters):
	caches = []
	A = X
	layers = len(parameters) // 2
	for layer in range(1, layers):
		A_pre = A
		A, cache= linear_activation_forward(A_pre, parameters["W"+str(layer)], parameters["b"+str(layer)], "relu")
		caches.append(cache)
	AL, cache = linear_activation_forward(A, parameters["W"+str(layers)], parameters["b"+str(layers)], "sigmoid")
	caches.append(cache)
	assert(AL.shape == (1,X.shape[1]))
	return AL, caches


def compute_cost(AL, Y, parameters):
    m = Y.shape[1]
    layers = len(parameters) // 2
    cost = np.sum(-Y*np.log(AL) - (1-Y)*np.log(1-AL))/m
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost

def linear_backward(dZ, cache):

	A_pre, W, b = cache
	m = dZ.shape[1]
	dW = np.dot(dZ,cache[0].T)/m
	db = np.sum(dZ, axis = 1, keepdims = True)/m
	dA_pre = np.dot(cache[1].T,dZ)
	assert(A_pre.shape == dA_pre.shape)
	assert(W.shape == dW.shape)
	assert(b.shape == db.shape)
	return dA_pre, dW, db

def linear_activation_backward(dA, cache, activation):

	linear_cache, activation_cache = cache
	if activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
		dA_pre, dW, db = linear_backward(dZ, linear_cache)
	elif activation == "relu":
		dZ = relu_backward(dA, activation_cache)
		dA_pre, dW, db = linear_backward(dZ, linear_cache)
	return dA_pre, dW, db

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0 
    assert (dZ.shape == Z.shape)
    return dZ

def model_backforward(AL, Y, caches, parameters):

	grads = {}
	m = AL.shape[1]
	Y = Y.reshape(AL.shape) 
	layers = len(caches)
	dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))
	current_cache = caches[layers-1] 

	grads["dA"+str(layers-1)], grads["dW"+str(layers)], grads["db"+str(layers)] = \
			linear_activation_backward(dAL, current_cache, "sigmoid")

	for layer in reversed(range(layers-1)):
	    current_cache = caches[layer]
	    grads["dA"+str(layer)], grads["dW"+str(layer+1)], grads["db"+str(layer+1)] = \
			linear_activation_backward(grads["dA"+str(layer+1)], current_cache, "relu")
        
	return grads