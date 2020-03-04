# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 22:29:22 2019

@author: mayur
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (4.0, 3.0)

# Preprocessing Input data
data = pd.read_csv('C:/Users/mayur/Desktop/csv1.csv')
X = data.iloc[:, 2]
Y = data.iloc[:, 4]
plt.scatter(X, Y)
plt.show()

m = 0
c = 0

L = 0.0001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
for i in range(len(X)): 
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    
print (m, c)

Y_pred = m*X + c

plt.scatter(X, Y) 
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()

