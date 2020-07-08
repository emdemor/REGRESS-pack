#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Description
----------
This module receives feature, target and error coluns and evaluates
the unidimensional linear regression. 

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

class linear:
	'''
	Description
	----------
	Instances of this class carry the properties
	of a simple linear regression.

	Arguments
	----------
	X (list or numpy.array): features for regression
	y (list or numpy.array): target values
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

	def __init__(self,X=[],y=[],y_errors=[]):

		# Fixing order since this just a polinomial regression of first order. LOL.
		# Users dont like complications
		order = 1

		# initializing status and self parameters
		self.parameters   = []
		self.par_estimate = []
		self.cov_matrix   = []
		self.par_error    = []

		# received parameters
		self.X		  = X
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
			self.y_errors= np.ones(len(self.X))

			self.remove_errors = True

		# The method onle accepts a bool for errors if it was False	
		else:
			self.remove_errors = False
			

		# Checking if the array parameters has the same size
		if(len(self.y)==self.n):
			# if it everithing all right, testing the shape
			# corrects if the shape is not right
			if(np.shape(self.X) == (self.n,1)):
				self.X = np.reshape(self.X,(-1,))
				#self.X = np.reshape(self.X,(-1,1))

			if(np.shape(self.y) == (self.n,1)):
				self.y = np.reshape(self.y,(-1,))

			if(np.shape(self.y_errors) == (self.n,1)):
				self.y_errors = np.reshape(self.y_errors,(-1,))

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

		# compacting variables
		x      = self.X
		f      = self.y
		errors = self.y_errors
		order  = self.order
		powers = np.arange(1+order)

		# local variables
		_X  = np.array(list(map(lambda n: np.divide(np.power(self.X,n),errors),powers)))
		_XT = np.array(list(map(lambda x: np.transpose(x),_X)))
		_Y  = np.divide(f,errors)
		M   = np.zeros((1+order,1+order))

		for i in range(1+order):
			for j in range(1+order):
				M[i,j] = 0.5*np.dot(_XT[i],_X[j])+0.5*np.dot(_XT[j],_X[i])

		V  = np.array(list(map(lambda xt: np.dot(xt,_Y),_XT)))
		MI = np.linalg.inv(M)

		# Evaluating properties
		self.cov_matrix    = MI
		self.par_estimate  = np.dot(MI,V)
		self.par_variance  = [MI[index,index] for index in range(1+order)]
		self.par_error     = [np.sqrt(MI[index,index]) for index in range(1+order)]
		self.fitted        = True


	def predict(self,X):
		'''
		Description
		----------
		After training, this method evaluates
		y for a given list of X

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
			powers = np.arange(1+self.order)
			k =self.par_estimate
			if type(X)==list:
				X = np.array(X)

			if (type(X) == int) or (type(X) == float):
				_sum_ = 0;
				for order in range(0,1+self.order):
					_sum_ += k[order] * X ** order
				return _sum_
			else:
				return np.sum(list(map(lambda n: np.multiply(k[n],np.power(X,n)),powers)),axis=0)


	def predict_error(self,X):
		'''
		Description
		----------
		After training, this method evaluates the error on
		y for a given X os a given list of X

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
			powers = np.arange(1+self.order)
			k = self.par_estimate
			s = self.par_error
			#print(s)

			if type(X)==list:
				X = np.array(X)

			if (type(X) == int) or (type(X) == float):
				_sum_ = 0;
				for order in range(0,1+self.order):
					_sum_ += (s[order] * X ** order) ** 2

				return _sum_ ** 0.5
			else:
				#return
				return np.sqrt(np.sum(list(map(lambda n: np.power(np.multiply(s[n],np.power(X,n)),2),powers)),axis=0))

	def plot(self,**kwargs):
		'''
		Description
		----------
		After training, this method returns a matplotlib graph
		comparing the dataset with the model

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
			#print("the keyword arguments are:", kwargs)
			opt = kwargs
			nModel = 20
			scatter_opt  =  {'s':		   20,
							 'c':		   'black',
							 'marker':	   '.',
							 'cmap':	   None,
							 'norm':		   None,
							 'vmin':		   None,
							 'vmax':		   None,
							 'alpha':	   None,
							 'linewidths': None,
							 'edgecolors': None,
							 'data':		   None,
							 'plotnonfinite':False,
							 'label'     : 'data'}

			line_opt = {'alpha' : 1,
				'color' :     'black',
				'linestyle' : '-',
				'linewidth' :  1,
				'label'     : 'model'}

			bars_opt = {'ecolor': 'black',
						'elinewidth': 0.75,
						'capsize': 2,
						'capthick': 0.75,
						'alpha': 1.,
						'ls':'none',
						'marker': '.',
						'markersize': None,
						'c': 'black',
						'label'     : 'data'}

			fill_opt = {'color': 'gray',
						'alpha': 0.2}

			for key in kwargs:
				if key in scatter_opt:
					scatter_opt[key] = kwargs[key]

				if key in line_opt:
					line_opt[key] = kwargs[key]

				if key in bars_opt:
					bars_opt[key] = kwargs[key]

			if 'linecolor' in kwargs:
				line_opt['color'] = kwargs['linecolor']

			if 'model_label' in kwargs:
				line_opt['label'] = kwargs['model_label']

			if 'data_label' in kwargs:
				bars_opt['label'] = kwargs['data_label']

			if 'lcolor' in kwargs:
				line_opt['color'] = kwargs['lcolor']

			if 'barcolor' in kwargs:
				bars_opt['ecolor'] = kwargs['barcolor']

			if 'bcolor' in kwargs:
				bars_opt['ecolor'] = kwargs['bcolor']

			if 'markercolor' in kwargs:
				bars_opt['c'] = kwargs['markercolor']

			if 'mcolor' in kwargs:
				bars_opt['c'] = kwargs['mcolor']

			if 'xlabel' in kwargs:
				xlabel = kwargs['xlabel']
			else:
				xlabel = 'x'

			if 'ylabel' in kwargs:
				ylabel = kwargs['ylabel']
			else:
				ylabel = 'y'

			if 'grid' in kwargs:
				grid = kwargs['grid']
			else:
				grid = False

			if 'fcolor' in kwargs:
				fill_opt['color'] = kwargs['fcolor']

			if 'falpha' in kwargs:
				fill_opt['alpha'] = kwargs['falpha']


			# if self.remove_errors:
			# 	bars_opt['capsize'] = 0
			# 	bars_opt['capthick'] = 0

			xmin = min(self.X)
			xmax = max(self.X)
			X_mod = np.linspace(xmin,xmax,nModel)

			plt.close()
			#
			

			if self.remove_errors:
				print('REMOVER ERROS: ', self.remove_errors)
				plt.scatter(self.X,self.y,**scatter_opt)

			else:
				plt.errorbar(x=self.X,y=self.y,yerr=self.y_errors,**bars_opt)

				plt.fill_between(X_mod,
					self.predict(X_mod)+self.predict_error(X_mod),
					self.predict(X_mod)-self.predict_error(X_mod),
					label='1$\sigma$ confidence region',**fill_opt)

			plt.plot(X_mod,self.predict(X_mod),**line_opt,zorder=10)

			plt.legend(loc="upper left")
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)

			#padx = 0.05
			#pady = 0.1
			#plt.xlim(xmin-padx*delta_X, xmax+padx*delta_X,)
			#plt.ylim(ymin-pady*delta_y, ymax+pady*delta_y)
			if grid:plt.grid()
			return plt