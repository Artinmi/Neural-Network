# -*- coding: utf-8 -*-
"""
sigmoid 1 hidden layer  meth1
@author: Artin
"""

import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Generate input and output data (sine wave)
x = np.linspace(-np.pi, np.pi, 400).reshape(400, 1)  # Reshape to 2D array
y = np.sin(x)

# Initialize weights with random values
weights1 = np.random.rand(1, 10)  # .rand() returns a randome float between 0 and 1
weights2 = np.random.rand(10, 1)

'''
weights1 = np.random.randint(-10, 10 , size=(1,10), dtype= float64 )  # .rand() returns a randome float between 0 and 1
weights2 = np.random.randint(-10, 10 , size=(10,1), dtype= float )
'''

# Learning rate and momentum
lr = 0.001
momentum = 0.9

prev_update_w1 = np.zeros_like(weights1)
prev_update_w2 = np.zeros_like(weights2)


for epoch in range(100000):
    # Forward pass
    layer0 = x
    layer1 = sigmoid(np.dot(layer0, weights1))
    layer2 = sigmoid(np.dot(layer1, weights2))

    # Backward pass (using the delta rule and momentum for weight updates)
    layer2_error = y - layer2
    layer2_delta = layer2_error * sigmoid_derivative(layer2) 
    
    layer1_error= np.dot(layer2_delta, weights2.T )
    layer1_delta = layer1_error * sigmoid_derivative(layer1)
    
    # Weight Update
    update_w2 = lr * np.dot(layer1.T,layer2_delta) + momentum * prev_update_w2
    update_w1 = lr * np.dot(layer0.T,layer1_delta) + momentum * prev_update_w1

    weights2 += update_w2
    weights1 += update_w1

    prev_update_w2 = update_w2
    prev_update_w1 = update_w1


# Plot the original sine wave and the network's output
plt.plot(x, y , label='Actual')
plt.plot(x, layer2 , label='Predicted')
plt.legend()
plt.savefig('sine wave via sigmoid', dpi=800)
