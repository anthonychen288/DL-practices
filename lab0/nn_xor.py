import numpy as np
from math import exp

def read_data(file):
	x = []
	with open(file, mode='r') as f:
		for line in f:
			x.append([int(x) for x in line.split(",")])

	return np.array(x)

def weightInit(input_units, layers=1):
	return 2 * np.random.random_sample((2, input_units)) - 1.0

def addBiasUnits(X=np.array([])):
	n_data = len(X)
	bias = np.ones((n_data, 1))
	return np.append(X, bias, axis=1)

def trainNN(X, y, epochs=10000):
	# Initialization
	n_hidden_layers = 1
	hidden_weights = weightInit(input_units=3, layers=n_hidden_layers)
	output_weights = 2.0 * np.random.random_sample(3) - 1.0
	# add bias to X
	n_data = len(X)
	design_x = addBiasUnits(X)
	# stores output of each hidden layers
	hidden_layer = np.ones(3)
	output = 0.0
	# learning rate
	r = 0.88
	last_change_output = np.zeros(3)
	last_change_hidden = np.zeros((2, 3))
	for iter in range(epochs):
		if (iter % 1000) == 0 and iter > 0:
			print("iteration\t%d" %(iter))
			#print("Error\t%f" %(error))

		
		for n in range(n_data):
			# feedforward
			# number of units to compute
			n_units = hidden_weights.shape[0]
			for j in range(n_units):
				hidden_layer[j+1] = sigmoid(design_x[n].dot(hidden_weights[j]))

			# output layer
			output = sigmoid(hidden_layer.dot(output_weights))
			# cost function
			error = ((y[n] - output)**2) / 2.
			if (iter % 1000) == 0 and iter > 0:
				print("error\t%.10f" %(error))
			# back propagation
			# update output layer
			n_output_units = len(output_weights)
			derivatives = np.zeros(n_output_units)
			for i in range(n_output_units):
				derivatives[i] +=  (output - y[n]) * gradient_sigmoid(output) * output_weights[i]
				output_weights[i] -=  r * (output-y[n]) * gradient_sigmoid(output) * hidden_layer[i]

			# update hidden layers
			for j in range(n_units):
				#tmp = gradient_sigmoid(output) * tmp_output_weights[j+1]
				for k in range(hidden_weights.shape[1]):
					hidden_weights[j][k] -= r * derivatives[j+1] * \
							gradient_sigmoid(hidden_layer[j+1]) * design_x[n][k]

			# adjust learning rate
			#r *= 0.88

	# final output
	print("\tground truth\toutput")
	for n in range(n_data):
		# feedforward
		n_units = hidden_weights.shape[0]
		for j in range(n_units):
			hidden_layer[j] = sigmoid(design_x[n].dot(hidden_weights[j]))

		# output layer
		output = sigmoid(hidden_layer.dot(output_weights))
		print(str(X[n])+"\t%f\t%.10f" %(y[n], output))

def sigmoid(x):
	return 1. / (1. + exp(x))

def gradient_sigmoid(x):
	return x * (1. - x)

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

trainNN(X=X, y=y)