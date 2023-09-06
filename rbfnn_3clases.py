# -*- coding: utf-8 -*-

#
# Author: Diana Valencia
# This code implements a Radial Basis Function Neural Network to 
# classify the iris flower data set.
#

import numpy as np
from sklearn import datasets
from scipy.cluster.vq import kmeans2
from numpy import linalg as ln
import math as mt


#
# Description: Calculate mu centers and sigma amplitudes.
# Parameters : data (z_p), Neurons in the hidden layer (J)
#
def compute_mu_and_sigma( z_p, J ):

	# Compute mu
	mu, idx = kmeans2(z_p, J, minit='++' )

	# Compute sigma
	sigma = np.zeros( J )
	for j in range( J ):
		d = []
		for k in range(J):
			#Compute the distance between mu_j y mu_k 
			d_k = ln.norm( mu[j] - mu[k] )
			d.append( d_k )
		delta = np.sort( d )
		sigma[j] = (delta[1]+delta[2])/2.0

	return mu, sigma


#
# Description: Calculates the Gaussian radial basis function
# Parameters : point z, center mu, radius sigma
#
def radial_basis_function( z, mu, sigma ):
	return mt.exp( -1*pow( ln.norm( z - mu ), 2.0)/( 2.0*sigma*sigma )  )

#
# Description: Computes the RBF of the set 
# Parameters: data set (data), centers (mu), radius (sigma)
#
def compute_rbf_in_set( data, mu, sigma, J ):
	P = len(data)
	phi = np.ones( (P, J + 1) )
	
	# Evaluate each training point in the RBF
	for p in range(P):
		for j in range(J):
			phi[p, j] = radial_basis_function( data[p], mu[j], sigma[j] ) 
			
	return phi

#
# Description: Train the neural network
# Parameters : Training set (data_training), neurons in the middle layer (J),
#			   desired output values(tarjet)
#
def train_network( data_training, J, target):
	# Compute centers and radius 
	mu, sigma = compute_mu_and_sigma( data_training, J )

	# Compute the RBF function 
	phi = compute_rbf_in_set( data_training, mu, sigma, J )

	# Estimate the weights
	w = np.matmul( np.matmul( ln.inv( np.matmul( phi.T, phi ) ), phi.T ), target)

	return w, mu, sigma

#
# Description : Convert each element of the array as follows:
#			   0 -> [1,0,0]	
#			   1 -> [0,1,0]
#			   2 -> [0,0,1]
# Parámetros  : An array with integer values ranging from 0 to 2 (target)
#
def target_to_3_outputs(target):
	new_target = []
	for t in target:
		if t == 0:
			new_target.append([1,0,0])
		elif t == 1:
			new_target.append([0,1,0])
		else:
			new_target.append([0,0,1])
	return new_target

#
# Description: Compute the accuracy of the testing set
# Parámetros : the output of the rbfnn (output) and the desired values (target)
#
def compute_accuracy(output, target):
	size_output = len(output)
	correct_patterns = 0.0
	for i in range( size_output ):
		if np.array_equal(output[i], target[i]): 
			correct_patterns += 1.0
	return correct_patterns/float(size_output)

def main():

	# Obtain the iris dataset 
	iris = datasets.load_iris()
	data = iris.data[:150]
	data_size = len(data)
	target = iris.target[:150]
	

	# Normalize data
	data = (data - data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))

	# Shuffle the data set
	idx = np.random.permutation(data_size)

	# Get the training set
	data_training = data[idx[:int(data_size*.75)]]
	target_training = target[idx[:int(data_size*.75)]]
	new_target_training = target_to_3_outputs(target_training)

	# Get the validation set
	data_validation = data[idx[int(data_size*.75):data_size]]
	target_validation = target[idx[int(data_size*.75):data_size]]
	new_target_validation = target_to_3_outputs(target_validation)

	J = 10
	# Train the neural network
	w, mu, sigma = train_network( data_training, J, new_target_training)

	# Validate neural network
	# Calculate the RBF for the validation set
	phi_validation = compute_rbf_in_set(data_validation, mu, sigma, J)

	# Get the output classes
	o = np.dot( phi_validation, w )

	for i in range( len(o) ):
		for j in range(3):
			if o[i,j] > 0.5:
				o[i,j] = 1
			else:
				o[i,j] = 0

	E = ln.norm( new_target_validation - o )
	acc = compute_accuracy(o, new_target_validation)
	print( "Er: ", E, " Accuracy:", acc )

	with open("out_rbfnn.txt",'w',encoding = 'utf-8') as f:
		f.write( " Target | RBFNN output \n" )
		for i in range( 25 ):
			f.write(  "".join([str(x) for x in new_target_validation[i]]) + " | " + "".join([str(int(x)) for x in o[i]]) +"\n"  )


if __name__ == "__main__":
    main()