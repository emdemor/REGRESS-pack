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

# Checking
X_obs = np.array([0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50])
y_obs = np.array([16.1,26.6,31.0,42.6,47.5,52.0,64.3,71.8,73.9,83.2,88.5])
y_err = np.array([2.00,2.00,2.10,2.20,2.20,2.30,2.30,2.40,2.40,2.50,2.50])

fit = rg.polynomial(X = X_obs,y = y_obs,y_errors = y_err,order=2)
X_mod = np.linspace(0.0,0.5,20)


fit.plot(xlabel='i (A)',ylabel='U (V)').savefig('teste.png')
