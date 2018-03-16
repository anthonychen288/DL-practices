import numpy as np
from math import exp

def read_data(file):
	x = []
	with open(file, mode='r') as f:
		for line in f:
			x.append([int(x) for x in line.split(",")])

	return np.array(x)

def weightInit(n_units, layers=1):
	return 2 * np.random.random_sample((layers, 2, n_units)) - 1.0

def addBiasUnits(X=np.array([])):
	n_data = len(X)
	bias = np.ones((n_data, 1))
	return np.append(X, bias, axis=1)

#def forwardFeed()

def train(X=np.array([]), hidden_weight=np.array([]), output_layer=np.array([]), epochs=1000):
	# add bias to X
	design_x = addBiasUnits(X)
	a = np.zeros((1, 3))
	n_hidden_layers = len(hidden_weight)
	output = 0.0

	for x in design_x:
		# forwardfeeding
		for i in range(n_hidden_layers):
			n_units = hidden_weight[i].shape[0]
			for j in range(n_units):
				a[i][j+1] = x.dot(hidden_weight[i][j].T)

			a[i] = sigmoid_vector(a[i])
		# output layer
		output = sigmoid(sum(a[-1] * output_layer))
		print output
		# cost function
		# back propagation

	#return output

def sigmoid(x):
	return 1. / (1. + exp(x))

def sigmoid_vector(x):
	return 1. / (1. + np.exp(x))

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# initialize random weights
# n_feats + 1(bias)
w_layers = weightInit(3, 1)
print("weights of hidden layers")
for weights in w_layers:
	print weights

w_output = 2.0 * np.random.random_sample(3) - 1.0
print("weights of output layer")
print w_output

train(X, w_layers, w_output)