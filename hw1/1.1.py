# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 09:13:09 2018

@author: MiaoWangqi

"""


from sklearn import datasets
from scipy import sparse
import numpy as np
import sklearn as sk
import datetime

#read file and convert to numpy array
filename = "D:/UCD_STA141C/hw1/cpusmall.txt"
X,y = datasets.load_svmlight_file(filename)
X_array = sparse.csr_matrix.todense(X) 
X_array = sk.preprocessing.normalize(X_array, axis=0)
y = y.reshape(y.size,1)
y = sk.preprocessing.normalize(y, axis=0)
#random omega vector initialize
vec_omega = np.random.rand(12).reshape(12,1)
lamda = 1
# change eta here
eta = 0.001
epsilon = 0.001



def L2Norm(vec):
    """
    @para: vec is 1D numpy array. 
    @return: a float.
    """
    return np.sqrt(vec.transpose().dot(vec))[0][0]

    
def GradientVec(X, omega, y, lamda):
    """
    X^T(Xw-y)
    @para: X is a 2D numpy array, omega and y is 1D numpy array.
    @return: 1D numpy array.
    """
    g = (X.transpose().dot(X.dot(omega)-y)*2/y.size + omega*lamda)
    return g        




r0 = L2Norm(GradientVec(X_array,vec_omega,y,lamda))
T1 = datetime.datetime.now()
iter = 0 
while (True):
     vec_omega = vec_omega-eta*GradientVec(X_array,vec_omega,y,lamda)
     iter += 1
     r = L2Norm(GradientVec(X_array,vec_omega,y,lamda))
     print(r)
     if( r < epsilon*r0 ):
         break
T2 = datetime.datetime.now()
T = (T2-T1).seconds
print("Time",T,"Iter",iter)
print(vec_omega)