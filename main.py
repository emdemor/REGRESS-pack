#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description
----------
Regress a simple python package to evaluate analitycal regression methods
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

dataset = pd.read_csv('examples/ex01_dataset.csv')
X     = dataset['X'].values.reshape(-1,1)
y     = dataset['y'].values.reshape(-1,1)
y_err = dataset['Erro'].values.reshape(-1,1)

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