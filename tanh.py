# -*- coding: utf-8 -*-
"""
@author: Artin
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# Generate a sine wave
x = np.arange(0, 2*np.pi, 0.1)
y = np.sin(x)

# Reshape data for MLPRegressor
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# Define the MLP with one hidden layer and tanh activation function
mlp = MLPRegressor(hidden_layer_sizes=(10,), activation='tanh', solver='sgd', learning_rate_init=0.01, momentum=0.9, max_iter=2000)

# Train the MLP
mlp.fit(x, y.ravel())

# Predict the sine wave
y_pred = mlp.predict(x)

# Plot actual vs predicted sine wave
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Actual')
plt.plot(x, y_pred, label='Predicted')
plt.legend()
plt.savefig('sine wave via tanh', dpi=800)