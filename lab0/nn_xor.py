import numpy as np
from math import exp

def addBiasUnit(X=np.array([])):
	n_data = len(X)
	bias = np.ones((n_data, 1))
	return np.append(X, bias, axis=1)

def sigmoid(x):
	return 1. / (1. + np.exp(-x))

def gradient_sigmoid(x):
	return x * (1. - x)

def trainNN(X, y, epochs=20000):
	# Initialization
	n_inputs, n_hidden_units, n_output_units = 2, 3, 1
	hidden_weights = np.random.uniform(low=-1.0, high=1.0, size=(2, n_inputs+1))
	output_weights = 2. * np.random.uniform(low=-1.0, high=1.0, size=3)
	# add bias to X
	n_data = len(X)
	design_x = addBiasUnit(X)
	# stores output of each hidden layers
	hidden_layer = np.ones(n_hidden_units)
	output = 0.0
	# learning rate
	r = 0.1

	for iter in range(epochs):

		cost = 0.0
		for n in range(n_data):

			# feedforward
			hidden_layer[:n_hidden_units-1] = sigmoid(design_x[n].dot(hidden_weights.T))
			# output layer
			output = sigmoid(hidden_layer.dot(output_weights))
			# cost function = (1/2)*SE
			cost += ((y[n] - output)**2 ) / 2.
			error = (y[n] - output)

			# back propagation
			# update output layer
			dCost_dh = np.ones(n_hidden_units)
			dCost_dh *= error * gradient_sigmoid(output) * output_weights
			output_weights += r * error * gradient_sigmoid(output) * hidden_layer
			# update hidden layers
			hidden_weights += r * np.outer(dCost_dh[:n_hidden_units-1] * \
				gradient_sigmoid(hidden_layer[:n_hidden_units-1]), design_x[n])

		if (iter%1000) == 0:
			print("iter: %d\trate: %f, cost: %f" %(iter, r, cost))
			
	# final output
	print("\tground truth\toutput")
	# feedforward
	H = addBiasUnit(sigmoid(design_x.dot(hidden_weights.T)))
	# output layer
	output = sigmoid(H.dot(output_weights))

	for i in range(n_data):
		print(str(X[i])+"\t%f\t%.10f" %(y[i], output[i]))

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])
for i in range(10):
	trainNN(X=X, y=y)
