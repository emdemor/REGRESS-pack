#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Description
----------
This module receives feature, target and error columns and evaluates
the multidimensional polynomial regression. 

Informations
----------
    Author: Eduardo M.  de Morais
    Maintainer:
    Email: emdemor415@gmail.com
    Copyright:
    Credits:
    License:
    Version:
    Status: in development

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from itertools   import product 


class multipoly:
	'''
	Description
	----------
	Instances of this class carry the properties
	of a polynomial regression.

	Arguments
	----------
	X (numpy.array): features for regression
	y (numpy.array): target values
	y_err (list, numpy.array or False): erros on target values
	data (bool): NOT IMPLEMENTED YET

	Parameters
	----------
	self.parameters (list): label model parameters
	self.par_estimate (numpy.array): estimate of parameters
	self.cov_matrix (numpy.matrix): covariance matrix of parameters
	self.cov_error (numpy.array): error in parameters
	self.X (numpy.array): features for regression
	self.y (numpy.array):  target values
	self.y_err (numpy.array): errors on targets
	self.fitted (bool): controls the fit approach
	self.order (int): not useful here
	self.n (int): number of data
	'''

	def __init__(self,
			X=np.array([]),
			y=np.array([]),
			y_errors=False,
			order = 2
		):

		# initializing status and self parameters
		self.parameters   = []
		self.parameters_matrix = []
		self.par_estimate = []
		self.cov_matrix   = []
		self.par_error    = []
		self.par_dim      = 0

		# received parameters
		self.X		  = X
		self.X_dim    = 0
		self.y        = y
		self.y_errors = y_errors
		self.fitted   = False
		self.order    = order
		self.n        = len(self.X)

		# The method onle accepts a bool for errors if it was False
		if type(self.y_errors) == bool:
			if self.y_errors:
				print('[Error] You must pass the values of errors')

			# choose a list of ones to errors variable	
			self.y_errors = np.ones(len(self.X))

			self.remove_errors = True

		# The method onle accepts a bool for errors if it was False	
		else:
			self.remove_errors = False
			
		print('X shape: ',self.X.shape)
		print('y shape: ',self.y.shape)

		# Checking if the array parameters has the same size
		if(len(self.y)==self.n):

			# corrects if the shape is not right
			if(len(np.shape(self.X)) != 2 ):
				print('[error] Features are not in the correct shape')
			else:
				self.X_dim = len(self.X[0])
				self.par_dim = int(np.sum([binom(self.X_dim+k-1,k) for k in range(0,self.order+1)]))

				print("ordem polinomial: ",self.order)
				print("qtd. atributos  : ",self.X_dim)
				print("parametr. livres: ",self.par_dim)

			if(np.shape(self.y) == (self.n,1)):
				self.y = np.reshape(self.y,(-1,))

			if(np.shape(self.y_errors) != (self.n,1)):
				print('[error] Target is not in the correct shape')

		else:
			print('[error]: Number of feature vectors, targets are not the same.')


	def train(self):
		'''
		Description
		----------
		Evalutes the properties of regression

		Arguments
		----------
		None

		Returns
		----------
		None

		'''

		# Evaluating the tensor variables
		list_Q,list_Q_comps = self.__get_tensor_structure()

		# First order
		if self.order == 1:
			# Obtaining the matrix structure of the linear system
			MI,V = self.__linear_system_struct_1(list_Q)

		# Order two
		if self.order == 2:
			# Obtaining the matrix structure of the linear system
			MI,V = self.__linear_system_struct_2(list_Q)

		# Solving the linear system for the parameter vector
		k_list = list(np.array(np.matmul(MI,V)).reshape(self.par_dim,))

		# Wrinting the parameter vector as a symmetric matrix
		k = np.zeros((1+self.order,1+self.order))


		# for i,j in product(range(self.X_dim),range(self.X_dim)):
		# 	if i>=j:
		# 		k[i,j] = k[j,i] = k_list[i+j]

		# Updating parameters
		# self.par_matrix    = k
		self.cov_matrix    = MI
		self.par_estimate  = k_list
		self.par_variance  = [MI[index,index] for index in range(self.par_dim)]
		self.par_error     = [np.sqrt(MI[index,index]) for index in range(self.par_dim)]
		self.fitted        = True


	def __get_tensor_structure(self):

		# List of auiliar structures of data
		list_Q = []
		list_Q_comps = []

		for dim_Q in range(1+2*self.order):

			if dim_Q == 0:
				Q = np.sum(1/np.power(self.y_errors,2))

			elif dim_Q == 1:
				shape_Q = (self.X_dim,)
				indexes_Q = [ele for ele in product(range(0, self.X_dim), repeat = dim_Q)]
				Q = np.zeros(shape_Q)
				for i in indexes_Q:
					Q_val = np.array([self.X[j,i] / (self.y_errors[j] ** 2) for j in range(self.n)]).sum()
					Q = self.__update_value_from_index(Q,i,Q_val)

			else:
				shape_Q = tuple([self.X_dim for i in range(dim_Q)])
				indexes_Q = [ele for ele in product(range(0, self.X_dim), repeat = dim_Q)] 
				Q = np.zeros(shape_Q)
				for index_list in indexes_Q:
					Q_val = np.array([np.array([self.X[j,i]/(self.y_errors[j]**2) for i in index_list]).prod() for j in range(self.n)]).sum()
					Q = self.__update_value_from_index(Q,index_list,Q_val)
			
			list_Q.append(Q)
			list_Q_comps.append(int(binom(self.X_dim + dim_Q-1,dim_Q)))

		return [list_Q,list_Q_comps]

	def __update_value_from_index(self,np_arr,indexes,new_value):
	    
	    s     = np_arr.size
	    shape = np_arr.shape
	    enum  = list(np.ndenumerate(np_arr))
	    sing_index = [elem[0] for elem in enum].index(indexes)
	    np_arr_temp = np_arr.reshape(s,)
	    np_arr_temp[sing_index] = new_value
	    return np_arr_temp.reshape(shape)


	def __part(self,np_arr,indexes):
	    
	    s     = np_arr.size
	    shape = np_arr.shape
	    enum  = list(np.ndenumerate(np_arr))
	    sing_index = [elem[0] for elem in enum].index(indexes)
	    np_arr_temp = np_arr.reshape(s,)
	    return np_arr_temp[sing_index]

	def __linear_system_struct_1(self,list_Q):
		if self.order == 1:
			A = list_Q[0]
			B = list_Q[1]
			C = list_Q[2]
			F = [np.array([self.y[j] / (self.y_errors[j] ** 2) for j in range(self.n)]).sum()]
			G = [np.array([self.y[j]*self.X[j,i] / (self.y_errors[j] ** 2) for j in range(self.n)]).sum() for i in range(self.X_dim)]

			# First matrix row
			m = [[0 for i in range(self.par_dim)] for j in range(self.par_dim)]
			m[0][0] = list_Q[0]
			for i in range(self.X_dim):
				m[0][1+i] = m[1+i][0] = B[i]
				for j in range(self.X_dim):
					m[1+i][1+j] = C[i][j]
			M = np.matrix(m)
			MI = np.linalg.inv(M)
			V = np.matrix((F+G)).T

			return [MI,V]
		else:
			return False



	def __linear_system_struct_2(self,list_Q):
		if self.order == 2:
			A = list_Q[0]
			B = list_Q[1]
			C = list_Q[2]
			D = list_Q[3]
			E = list_Q[4]
			F = [np.array([self.y[j] / (self.y_errors[j] ** 2) for j in range(self.n)]).sum()]
			G = [np.array([self.y[j]*self.X[j,i] / (self.y_errors[j] ** 2) for j in range(self.n)]).sum() for i in range(self.X_dim)]
			H = list([np.array([self.y[j]*self.X[j,i]*self.X[j,k] / (self.y_errors[j] ** 2) for j in range(self.n)]).sum() for i,k in product(range(self.X_dim),range(self.X_dim)) if i<=k])

			# First matrix row
			m = [[0 for i in range(self.par_dim)] for j in range(self.par_dim)]
					
			m[0][0] = list_Q[0]
			count_C = 0
			
			for i in range(self.X_dim):

				m[0][1+i] = m[1+i][0] = B[i]
				for j in range(self.X_dim):
					if i<=j:
						m[0][1+self.X_dim+count_C] = m[1+self.X_dim+count_C][0] = C[i][j]

						count_E = 0

						for k in range(self.X_dim):
							m[1+k][1+self.X_dim+count_C] = m[1+self.X_dim+count_C][1+k] = D[k][i][j]

							for l in range(self.X_dim):

								if k <= l:
									m[1+self.X_dim+count_E][1+self.X_dim+count_C] = E[k][l][i][j]
									count_E += 1

						# m[1+self.X_dim+count_E][1+self.X_dim+count_C] = E[0][0][i][j]
						
						count_C += 1

					m[1+i][1+j] = C[i][j]


			M = np.matrix(m)

			MI = np.linalg.inv(M)

			V = np.matrix((F+G+H)).T

			return [MI,V]
		else:
			return False


