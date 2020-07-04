#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Description
----------
???????????????????????

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

	def __init__(self,X=[],y=[],y_errors=[],data=False):

		# Fixing order since this just a polinomial regression of first order. LOL.
		# Users dont like complications
		order = 1

		# initializing status and self parameters
		__STATUS_OK__ = True
		self.parameters = []
		self.par_estimate  = []
		self.cov_matrix = []
		self.par_error  = []

		# received parameters
		self.X		    = X
		self.y          = y
		self.y_errors   = y_errors
		self.fitted     = True
		self.order      = order
		self.n          = len(self.X)


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
			__STATUS_OK__ = False


		# compacting variable labels
		x = self.X

		f = self.y

		errors = self.y_errors

		powers = np.arange(1+self.order)

		_X  = np.array(list(map(lambda n: np.divide(np.power(self.X,n),errors),powers)))

		_XT = np.array(list(map(lambda x: np.transpose(x),_X)))

		_Y = np.divide(f,errors)

		M = np.zeros((1+order,1+order))

		for i in range(1+order):
			for j in range(1+order):
				M[i,j] = 0.5*np.dot(_XT[i],_X[j])+0.5*np.dot(_XT[j],_X[i])

		V = np.array(list(map(lambda xt: np.dot(xt,_Y),_XT)))

		MI = np.linalg.inv(M)

		self.cov_matrix = MI

		self.par_estimate  = np.dot(MI,V)

		self.par_variance = [MI[index,index] for index in range(1+order)]
		self.par_error    = [np.sqrt(MI[index,index]) for index in range(1+order)]


	def predict(self,X):
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
		#print("the keyword arguments are:", kwargs)
		opt = kwargs
		nModel = 20
		scatter_opt  =  {'s':		   None,
						 'c':		   None,
						 'marker':	   None,
						 'cmap':	   None,
						 'norm':		   None,
						 'vmin':		   None,
						 'vmax':		   None,
						 'alpha':	   None,
						 'linewidths': None,
						 'edgecolors': None,
						 'data':		   None,
						 'plotnonfinite':False }

		line_opt = {'alpha' : 1,
			'color' :     'black',
			'linestyle' : '-',
			'linewidth' :  1,
			'label'     : 'model'}

		bars_opt = {'ecolor': 'black',
					'elinewidth': 0.75,
					'capsize': 2,
					'capthick': 0.75,
					'alpha': 1,
					'ls':'none',
					'marker': '.',
					'markersize': None,
					'c': 'black',
					'label'     : 'data'}

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

		if self.remove_errors:
			bars_opt['capsize'] = 0
			bars_opt['capthick'] = 0

		xmin = min(self.X)
		xmax = max(self.X)
		X_mod = np.linspace(xmin,xmax,nModel)

		plt.close()
		#plt.scatter(self.X,self.y,**scatter_opt)
		plt.errorbar(x=self.X,y=self.y,yerr=self.y_errors,**bars_opt)
		plt.fill_between(X_mod,
			self.predict(X_mod)+self.predict_error(X_mod),
			self.predict(X_mod)-self.predict_error(X_mod),
			color='gray', alpha=0.2,label='1$\sigma$ confidence region')
		plt.plot(X_mod,self.predict(X_mod),**line_opt)

		plt.legend(loc="upper left")
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		if grid:plt.grid()
		return plt