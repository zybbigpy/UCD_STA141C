# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 09:13:09 2018

@author: MiaoWangqi

"""

#%%
from sklearn import datasets
from scipy import sparse
import numpy as np

#read file and convert to numpy array
filename = "D:/UCD_STA141C/hw1/cpusmall.txt"
X,y = datasets.load_svmlight_file(filename)
X_array = sparse.csr_matrix.todense(X) 
X_array.shape
y=y.reshape(y.size,1)
#random omega vector initialize
vec_omega = np.random.rand(12).reshape(12,1)
lamda = 1
eta = 0.00000000000000000001
#%%

#%%
def L2Norm(vec):
    """
    @para: vec is 1D numpy array. 
    @return: a float.
    """
    return np.sqrt(vec.transpose().dot(vec))[0][0]

#%%
    

#%%
def GradientVec(X, omega, y, lamda):
    """
    X^T(Xw-y)
    @para: X is a 2D numpy array, omega and y is 1D numpy array.
    @return: 1D numpy array.
    """
    g = (X.transpose().dot(X.dot(omega)-y)*0.5/y.size + omega*lamda).A
    return g        
#%%


#%%

for i in range (5000):
    vec_omega = vec_omega-eta*GradientVec(X_array,vec_omega,y,lamda)
    
print(vec_omega)
#%%











