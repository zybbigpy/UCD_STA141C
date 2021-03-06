# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 14:13:02 2018

@author: MiaoWangqian
"""


from sklearn import datasets
from scipy import sparse
import numpy as np
filename="D:/UCD_STA141C/hw1/E2006.train.bz2"
X,y = datasets.load_svmlight_file(filename)
X = X[:,:-2]
y = sparse.csc_matrix(y).T
omega = np.random.rand(150358).reshape(150358,1)
omega = sparse.csc_matrix(omega)
lamda = 1 




def Gradient(X, y, omega, lamda):
    
    g = (X.T@(X@omega-y)*2/y.shape[0] + omega*lamda)
    
    return g    



def Norm(vec):
    
    return np.sqrt(vec.T@vec)[0,0]



r0 = Norm(Gradient(X,y,omega,lamda))
epsilon = 0.001
eta = 0.01
#learn oemga
while (True):
     omega = omega-eta*Gradient(X,y,omega,lamda)
     r = Norm(Gradient(X,y,omega,lamda))
     print(r)
     if( r < epsilon*r0 ):
         break     

         

def MSE(X,omega,y):
    return Norm(X@omega-y)/y.shape[0]

        
        
     
# Test on the dataset and get MSE 
TestFilename = "D:/UCD_STA141C/hw1/E2006.test.bz2"         
X_test,y_test = datasets.load_svmlight_file(TestFilename)
y_test = sparse.csc_matrix(y_test).T   
  

mse = MSE(X_test,omega,y_test)
print(mse)



