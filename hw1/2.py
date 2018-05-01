# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 15:29:57 2018

@author: MiaoWangqian
"""


from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy import sparse
import numpy as np

filename="D:/UCD_STA141C/hw1/news20.binary.bz2"
X,y = datasets.load_svmlight_file(filename)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train = sparse.csr_matrix(y_train).T
y_test = sparse.csr_matrix(y_test).T
omega = np.random.randn(1355191).reshape(1355191,1)
omega = sparse.csr_matrix(omega)
lamda = 1

#gradient
def h(x):
    return 1/(1+np.exp(x))

def Gradient(X,y,omega,lamda):
    a = np.array(sparse.csr_matrix.todense(X@omega))
    b = np.array(sparse.csr_matrix.todense(y))
    g =  X.T@sparse.csr_matrix(-b*h(b*a))/y.shape[0]+ lamda*omega
    return g
 
    

def Norm(vec):
    
    return np.sqrt(vec.T@vec)[0,0]

    
# train
r0 = Norm(Gradient(X_train,y_train,omega,lamda))
epsilon=0.001
eta =0.01
while (True):
     omega = omega-eta*Gradient(X_train,y_train,omega,lamda)
     r = Norm(Gradient(X_train,y_train,omega,lamda))
     print(r)
     if( r < epsilon*r0 ):
         break     


     
# test
n = 0
y_star = X_test@omega
for i in range(y_test.shape[0]):
    if (y_star[i]*y_test[i] > 0):
        n += 1;
print(n/y_test.shape[0])        

         
