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
from .tools import _input_data

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
			X         = False,
			y         = False,
			dataframe = False,
	        features  = False,
	        target_error = False,
	        target    = False,
			y_error  = False,
			order     = 2
		):

		results = _input_data(X = X,
	           				  y = y, 
	           				  y_error = y_error,
	           				  dataframe = dataframe,
	        				  features  = features,
	                          target_error = target_error,
	                          target    = target
	           				  )

		if not results['status']:
			print('[error]: Unable to import dataset.')

		else:

			# initializing status and self parameters
			self.parameters   = []
			self.parameters_matrix = []
			self.par_estimate = []	
			self.cov_matrix   = []
			self.par_error    = []
			self.par_dim      = 0


			# received parameters
			self.X		  = results['X']
			self.X_dim    = results['n_X']  
			self.y        = results['y']
			self.y_error  = results['y_error']
			self.remove_errors = not results['errors']
			self.n        = len(self.X)
			self.fitted   = False
			self.order    = order

			self.par_dim = int(np.sum([binom(self.X_dim+k-1,k) for k in range(0,self.order+1)]))

			print("Polynomial order  : ",self.order)
			print("Features          : ",self.X_dim)
			print("Degrees of Freedom: ",self.par_dim)



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
		#k = np.zeros((1+self.order,1+self.order))

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


	def predict(self,X):
		'''
		Description
		----------
		After training, this method evaluates
		y for a list of X

		Arguments
		----------
		Not documented yet

		Returns
		----------
		Not documented yet
		
		'''
		if not self.fitted:
			print('[error]: You must train first.')

		else:
			if True:
				X = np.array(X)
				k = self.par_estimate
				n = len(X)
				_sum_ = k[0]

				if np.shape(X) == (self.X_dim,):
					X = np.array([X])
					n = 1

				for i in range(self.X_dim):
					_sum_ += k[1+i]*X[:,i]

				if self.order >= 2:
					count = 1+self.X_dim
					for i,j in product(range(self.X_dim),range(self.X_dim)):
						if i<= j:
							_sum_ += k[count]*X[:,i]*X[:,j]
							count += 1

				#print(_sum_,' ',n)

				if n == 1:
					result = np.array(_sum_)
				else:
					result = np.array(_sum_).reshape(n,1)

				return result




	def __get_tensor_structure(self):
		'''
		Description
		----------
		Generates the tensor structures used to evaluate
		the matrix.

		Arguments
		----------
		Not documented yet

		Returns
		----------
		list_Q (list): a list containing the numpy.array structures constructed with the features an errors
		list_Q_comps (list): a list with the size of each structure
		
		'''

		# List of auxiliar structures of data
		list_Q = []
		list_Q_comps = []

		for dim_Q in range(1+2*self.order):

			if dim_Q == 0:
				Q = np.sum(1/np.power(self.y_error,2))

			elif dim_Q == 1:
				shape_Q = (self.X_dim,)
				indexes_Q = [ele for ele in product(range(0, self.X_dim), repeat = dim_Q)]
				Q = np.zeros(shape_Q)
				for i in indexes_Q:
					Q_val = np.array([self.X[j,i] / (self.y_error[j] ** 2) for j in range(self.n)]).sum()
					Q = self.__update_value_from_index(Q,i,Q_val)

			else:
				shape_Q = tuple([self.X_dim for i in range(dim_Q)])
				indexes_Q = [ele for ele in product(range(0, self.X_dim), repeat = dim_Q)] 
				Q = np.zeros(shape_Q)
				for index_list in indexes_Q:
					Q_val = np.array([np.array([self.X[j,i]/(self.y_error[j]**2) for i in index_list]).prod() for j in range(self.n)]).sum()
					Q = self.__update_value_from_index(Q,index_list,Q_val)
			
			list_Q.append(Q)
			list_Q_comps.append(int(binom(self.X_dim + dim_Q-1,dim_Q)))

		return [list_Q,list_Q_comps]




	def __update_value_from_index(self,np_arr,indexes,new_value):
		'''
		Description
		----------

		Arguments
		----------
		Not documented yet

		Returns
		----------
		Not documented yet
		
		'''
		s     = np_arr.size
		shape = np_arr.shape
		enum  = list(np.ndenumerate(np_arr))
		sing_index = [elem[0] for elem in enum].index(indexes)
		np_arr_temp = np_arr.reshape(s,)
		np_arr_temp[sing_index] = new_value
		return np_arr_temp.reshape(shape)




	def __part(self,np_arr,indexes):
		'''
		Description
		----------
		Receives a array and a a list (or a tuple) returning the related component.

		Arguments
		----------
		np_arr (numpy.array): a general numpy array
		indexes (list or tuple): a list or tuple describing an elemento of np_arr

		Returns
		----------
		(array type): element of np_arr in the
		
		'''
		
		s     = np_arr.size
		shape = np_arr.shape
		enum  = list(np.ndenumerate(np_arr))
		sing_index = [elem[0] for elem in enum].index(indexes)
		np_arr_temp = np_arr.reshape(s,)
		return np_arr_temp[sing_index]




	def __linear_system_struct_1(self,list_Q):
		'''
		Description
		----------

		Arguments
		----------
		Not documented yet

		Returns
		----------
		Not documented yet
		
		'''
		if self.order == 1:
			A = list_Q[0]
			B = list_Q[1]
			C = list_Q[2]
			F = [np.array([self.y[j] / (self.y_error[j] ** 2) for j in range(self.n)]).sum()]
			G = [np.array([self.y[j]*self.X[j,i] / (self.y_error[j] ** 2) for j in range(self.n)]).sum() for i in range(self.X_dim)]

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
		'''
		Description
		----------

		Arguments
		----------
		Not documented yet

		Returns
		----------
		Not documented yet
		
		'''
		if self.order == 2:
			A = list_Q[0]
			B = list_Q[1]
			C = list_Q[2]
			D = list_Q[3]
			E = list_Q[4]
			F = [np.array([self.y[j] / (self.y_error[j] ** 2) for j in range(self.n)]).sum()]
			G = [np.array([self.y[j]*self.X[j,i] / (self.y_error[j] ** 2) for j in range(self.n)]).sum() for i in range(self.X_dim)]
			H = list([np.array([self.y[j]*self.X[j,i]*self.X[j,k] / (self.y_error[j] ** 2) for j in range(self.n)]).sum() for i,k in product(range(self.X_dim),range(self.X_dim)) if i<=k])

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


