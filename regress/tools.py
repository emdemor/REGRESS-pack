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
import pandas as pd

def _input_data(X=False,
	           y=False, 
	           y_error=False,
	           dataframe = False,
	           features = False,
	           target_error = False,
	           target = False
	           ):
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
	__STATUS__ = True
	__TYPES__ = (list,
	             np.ndarray,
	             np.generic,
	             pd.core.frame.DataFrame
	             )
	__LEN__ = 0
	__MSG__ = ''
	__ERRORS__ = False
	# __XY__  = False
	# __DF__  = False

	# Checking if arguments was passed
	if __STATUS__ & (isinstance(X,bool) or isinstance(y,bool)) & isinstance(dataframe,bool):
		__MSG__ = __MSG__ +'[error]: Pass both X and y or a dataframe containing both.'
		__STATUS__ &= False

	# Checking type of X
	if __STATUS__ & (not isinstance(X,__TYPES__)) & (not isinstance(X,bool)):
		__MSG__ = __MSG__ +'[error]: Type of parameter "X" is not right'
		__STATUS__ &= False
	
	# Checking type of y
	if __STATUS__ & (not isinstance(y,__TYPES__)) & (not isinstance(y,bool)):
		__MSG__ = __MSG__ +'[error]: Type of parameter "y" is not right'
		__STATUS__ &= False
	
	# Checking type of y_error
	if __STATUS__ & (not isinstance(y,__TYPES__)) & (not isinstance(y,bool)):
		__MSG__ = __MSG__ +'[error]: Type of parameter "y_error" is not right'
		__STATUS__ &= False
	     
	# Checking type of dataframe
	if __STATUS__ & (not isinstance(dataframe,(pd.core.frame.DataFrame,bool))):
		__MSG__ = __MSG__ +'[error]: Type of argument "dataframe" is not right'
		__STATUS__ &= False




	# Or X and y are lists, or dataframe is a pandas.DataFrame  
	if __STATUS__ & ((isinstance(X,__TYPES__) & isinstance(y,__TYPES__) ) or isinstance(dataframe,pd.core.frame.DataFrame)):

		# if user passes all arguments, the code will prefer work with X and y rather than dataframe
		if isinstance(X,__TYPES__) & isinstance(y,__TYPES__) & isinstance(dataframe,pd.core.frame.DataFrame):
			__MSG__ = __MSG__ +'\n[warning]: All arguments "X", "y" and "dataframe" was passed. Preference will be given to X and y.'
			dataframe = False

		# If user dont pass X or y, but pass dataframe, uses just dataframe
		if isinstance(dataframe,pd.core.frame.DataFrame) & ( (not isinstance(X,__TYPES__)) or (not isinstance(y,__TYPES__)) ):
			__MSG__ = __MSG__ +'\n[warning]: Arguments "X" or "y" was not passed. Preference will be given to  "dataframe".'
			X = False
			y = False

	elif __STATUS__ & isinstance(dataframe,(pd.core.frame.DataFrame,bool)):
		__MSG__ = __MSG__ +'\n[error]: Some of the arguments "X", "y" or "dataframe" are not in the correct types.'
		__STATUS__ &= False

	# # If X is a list, array or dataframe, get the size
	# if __STATUS__ & (not isinstance(X,__TYPES__)):
	# 	__LEN__ = len(X)


	# When the user pass a dataframe 
	if __STATUS__ & (not isinstance(dataframe,bool)):
		
		# getting the size of dataframe
		__LEN__ = len(dataframe)

		# if the user doesnt estipulate tha target, assumes that it is the last
		if isinstance(features,list) & isinstance(target,str):
			X = dataframe[features].to_numpy()
			y = dataframe[target].to_numpy().reshape(-1,1)

			if isinstance(target_error,str):
				y_error = dataframe[target].to_numpy().reshape(-1,1)
				__ERRORS__ = True
			else:
				y_error = np.ones(__LEN__).reshape(-1,1)

		elif (not isinstance(features,list)) & isinstance(target,str):
			# get a list of columns
			features = list(dataframe.columns)

			# removing the target
			features.remove(target)

			# interpreting error
			if isinstance(target_error,str):
				features.remove(target)
				y_error = dataframe[target_error].to_numpy().reshape(-1,1)
				__ERRORS__ = True
			else:
				y_error = np.ones(__LEN__).reshape(-1,1)

			X = dataframe[features].to_numpy()
			y = dataframe[target].to_numpy().reshape(-1,1)

		elif  isinstance(features,bool) &  isinstance(target,bool):
			if (features == False) & (target == False ):
				target = list(dataframe.columns)[-1]
				features = list(dataframe.columns)
				features.remove(target)
				X = dataframe[features].to_numpy()
				y = dataframe[target].to_numpy().reshape(-1,1)
				y_error = np.ones(__LEN__).reshape(-1,1)


	elif __STATUS__ & isinstance(X,__TYPES__) & isinstance(y,__TYPES__):

		if __STATUS__  &  (len(X) != len(y)):
			__MSG__ = __MSG__ +'\n[error]: The arguments "X" and "y" are not the same size.'
			__STATUS__ = False

		elif __STATUS__  &  (len(X) == len(y)):

			__LEN__ = len(X)

			# if X is a list, reshape and transform it into numpy

			if __STATUS__ & isinstance(X,list):
				X = np.array(X)

			if __STATUS__ & isinstance(y,list):
				y = np.array(y)

			if __STATUS__ & (X.shape == (__LEN__,)):
				X = X.reshape(-1,1)

			if __STATUS__ & (y.shape == (__LEN__,)):
				y = y.reshape(-1,1)

			if __STATUS__ & isinstance(y_error,__TYPES__):
				if __STATUS__ & (y_error.shape == (__LEN__,)):
					y_error = np.array(y_error)
					y_error = y_error.reshape(-1,1)
					__ERRORS__ = True


				if __STATUS__ & (len(y_error) != __LEN__):
					__MSG__ = __MSG__ +'\n[error]: The arguments "X", "y" and "y_error" are not the same size.'
					__STATUS__ = False



			elif __STATUS__ & isinstance(y_error,bool):
				if __STATUS__ & y_error == False:
					y_error = np.ones(__LEN__).reshape(-1,1)

	if __STATUS__:
		results = {'X': X,
		           'y': y,
		           'y_error': y_error,
		           'log': __MSG__,
		           'n_data': __LEN__,
		           'n_X': (X.shape)[1],
		           'status':True,
		           'errors': __ERRORS__
		           }
	else:
		results = {'X':None,'y':None,'y_error':None,'log':None,'n_data':None,'n_X':None,'status':False,'errors':None}

	return  results

	     
	    
	     
	
	     
	     