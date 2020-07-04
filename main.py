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

dataset = pd.read_csv('data/data_test.csv')

X     = dataset['X'].values.reshape(-1,1)
y     = dataset['y'].values.reshape(-1,1)
y_err = dataset['Erro'].values.reshape(-1,1)


fit = rg.linear(X = X,y = y, y_errors = y_err)
y_pred = fit.predict(X)

fit.plot(xlabel='i (A)',ylabel='U (V)',color='red').savefig('linear_fit.png')


print(fit.predict_error(1))
print(fit.predict_error([1,2]))

# print('linear parameters: ',fit_lin.estimates)

#fit_quad = rg.polynomial(X = X,y = y, y_errors = y_err,order=2)
#fit_quad.plot(xlabel='i (A)',ylabel='U (V)',color='red').savefig('quad_fit.png')
#print('quadratic parameters: ',fit_quad.estimates)


# fit_lin = rg.linear(X = X,y = y, y_errors = False)
# fit_lin.plot(xlabel='i (A)',ylabel='U (V)',color='red').savefig('linear_fit.png')
# print('linear parameters: ',fit_lin.estimates)