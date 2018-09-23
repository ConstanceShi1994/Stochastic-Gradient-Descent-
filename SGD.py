
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd
import random
## Derive the stochastic gradient descent 

"""
batch size - hyperparameters you'll be tuning when you train neural network with
mini-batch SGD and is data dependent. 
"""

# Load data
x = pd.read_csv("X_train.csv", sep = ',')
y = pd.read_csv("Y_train.csv", sep = ',')
x_train = pd.DataFrame(x)
x_train.loc[5331] = x_train.columns
y_train = pd.DataFrame(y)
y_train.loc[5331] = y_train.colimns
"""
Create columns:
age: subject age in years
sex: subject gender, '0' - male, '1' - female
Jitter: Several measures of variations in fundamental frequency of voice
Shimmer: Several measures of variations in fundamental amplitude of voice
NHR, HNR: two measures of ratio of noise to tonal components in the voice
RPDE: a nonlinear dynamical complexity measure
DFA: signal fractal scaling exponent
PPE: a nonlinear measure of fundamental frequency variation
"""
x_train.columns = ['age', 'sex', 'Jitter(%)', 'Jitter(abs)','Jitter(RAP)', 'Jitter(PPQ5)','Jitter(DDP)','Shimmer','Shimmer(dB)','Shimmer(APQ3)','Shimmer(APQ5)','Shimmer(APQ11)','Shimmer(DDA)','NHR','HNR','RPDE','DFA','PPE']
x_train.head()
num = y_train.shape[0]

%matplotlib inline
# Initialize the parameters
def init_params():
	# Initialize beta with random normal values
	beta = np.random.normal(scale = 1, size = (x_train.shape[1],1))
	beta_0 = np.zeros(shape(1,))
	params = [beta, beta_0]
	return params 

# Stochastic gradient descent 
def sgd(params, t_k, batch_size):
	for param in params:
		param[:] = param - t_k*param/batch_size


# Construct data iterator 
def data_iter(batch_size):
	idx = list(range(x_train.shape[0]))
	random.shuffle(idx)
	for batch_i, i in enumerate(range(0, x_train.shape[0], batch_size)):
		j = np.array(idx[i:min(i+batch_size, x_train.shape[0])])
		yield batch_i, x_train.loc[j], y_train.loc[j]

# Linear regression
def linear_regression(X, beta, beta_0):
	return np.dot(X, beta) + beta_0

# Loss function
def square_loss(yhat, y, batch_size, beta):
	return (yhat - y) ** 2 /(2*batch_size) + (1/2)*np.linalg.norm(beta, ord=2)**2

# Train
def train(batch_size, t_k, epochs, period):
	beta, beta_0 = init_params()
	total_loss = []
	# Epoch starts from 1
	for epoch in range(1, epochs+1):
		for batch_i, data, label in data_iter(batch_size):
			with autograd.record():
				output = linear_regression(data, beta, beta_0)
				loss = square_loss(output, label, batch_size, np.append(beta,beta_0))
			loss[::-1]
			sgd([beta, beta_0], t_k, batch_size)
			if batch_i * batch_size % period == 0:
				total_loss.append(np.mean(square_loss(linear_regression(data, beta, beta_0), label, batch_size, np.append(beta,beta_0)))
	print("Batch size %d, Learning rate %f, Epoch %d, loss %.4e" % (batch_size, t_k, epoch, total_loss[-1]))
	x_axis = np.linspace(0, epochs, len(total_loss), endpoint=True)
    plt.semilogy(x_axis, total_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    
for i in (10, 20, 50, 100):
	for j in (10**(-2), 10**(-3), 10**(-4), 10**(-5)):
		train(batch_size = i, t_k = j, epochs = 500, period = x_train.shape[0])

## Implement the stochatic gradient descent algorithm to solve ridge regression
# Initialize beta with random normal values

# Fit the model parameters on the training data 

# Evaluate the objective function after each epoch

## Plot f^(k) - f^(*) versus k on a semi-log scale 
