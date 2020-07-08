#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description
----------
Regress is a simple python package to evaluate analitycal regression methods
for model linear on its parameters.

Informations
----------
    Author: Eduardo M.  de Morais
    Maintainer:
    Email: emdemor415@gmail.com
    Copyright:
    Credits:
    License:
    Version:
    Status: in intial stage of development
"""

import regress as rg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import binom

dataset = pd.read_csv('examples/ex01_dataset.csv')
X     = dataset['X'].values.reshape(-1,1)
y     = dataset['y'].values.reshape(-1,1)
y_err = dataset['Erro'].values.reshape(-1,1)

if False:
    filenames = []
    for order in range(0,11):
        fit = rg.polynomial(X = X,y = y, y_errors = y_err,order=order)
        fit.train()
        filename = 'examples/ex01/ex01_order_'+str(order).zfill(2)+'.png'
        #filenames.append(filename)
        fit.plot(xlabel='i (A)',
                  ylabel='U (V)',
                  color='red',
                  nPoints=100,
                  model_label='regression - order '+str(order)).savefig(filename)



D = 3
O = 2


if D == 1:
    dataset = pd.read_csv('examples/ex01_dataset.csv')
    X     = dataset['X'].values.reshape(-1,1)
    y     = dataset['y'].values.reshape(-1,1)
    y_err = dataset['Erro'].values.reshape(-1,1)
    fit = rg.multipoly(X,y,y_err,order=O)
    fit.train()

if D == 2:

    dataset = pd.read_csv('examples/ex03_multilin_dataset.csv')
    X = dataset.iloc[:,0:-2].values
    y = dataset.iloc[:,2].values.reshape(-1,1)
    y_err =  dataset.iloc[:,3].values.reshape(-1,1)
    fit = rg.multipoly(X,y,y_errors = y_err,order=O)
    fit.train()

if D == 3:
    dataset = pd.read_csv('examples/ex04_multilin_3.csv')
    X = dataset[['X','Y','Z']].values
    y = dataset[['F']].values
    y_err =  dataset[['Error']].values
    fit = rg.multipoly(X,y,y_err,order=O)
    fit.train()
    #print(y_err)
    

#print(X)
#print(np.sum(np.power(fit.predict(X)-y,2)))
#print(fit.predict(X[-1]),' ',y[-1])
#print(fit.predict(X))


# D = 3;
# O = 2;

# sum_ = 1
# for k in range(1,O+1):
#     sum_ += binom(D+k-1,k)
# print('order:',O,'    features:',D,'   nP:',sum_)


# print(np.sum([binom(D+k-1,k) for k in range(0,O+1)]))

# teste = np.array([
#     [[0,1,2],[3,4,5],[6,7,8]],
#     [[9,10,11],[12,13,14],[15,16,17]],
#     [[18,19,20],[21,22,23],[24,25,26]]
#     ])

# indexes = (1,1,1)

# s = teste.size
# en = list(np.ndenumerate(teste))

# sing_index = [elem[0] for elem in en].index(indexes)

# sh = teste.shape
# teste2 = teste.reshape(s,)

# teste2[sing_index] = 42

# teste = teste2.reshape(sh)


#from itertools import product 

#teste = np.array([i for i in product(range(1,6), range(1,6))])

# def __change(np_arr,indexes,new_value):
    
#     s     = np_arr.size
#     shape = np_arr.shape
#     enum  = list(np.ndenumerate(np_arr))
#     sing_index = [elem[0] for elem in enum].index(indexes)
#     np_arr_temp = np_arr.reshape(s,)
#     np_arr_temp[sing_index] = new_value
#     return np_arr_temp.reshape(shape)
    
# teste3 = __change(teste,(1,1,1),23)