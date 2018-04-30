# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 15:29:57 2018

@author: MiaoWangqian
"""

#%%
from sklearn import datasets
from scipy import sparse
import numpy as np

filename="D:/UCD_STA141C/hw1/news20.binary.bz2"
X,y = datasets.load_svmlight_file(filename)
y = sparse.csr_matrix(y).T
omega = np.random.rand(1355191).reshape(1355191,1)
omega = sparse.csr_matrix(omega)
lamda = 1
#%%

#%%
def Gradient(X,y,omega,lamda):

    g = -X.T@y/(1+np.exp(((omega.T)@(X.T)@y)[0,0])) +omega
    return g
#%% 
    
#%%
def Norm(vec):
    
    return np.sqrt(vec.T@vec)[0,0]
#%%
    
#%%
r0 = Norm(Gradient(X,y,omega,lamda))
epsilon=0.001
eta =0.01
while (True):
     omega = omega-eta*Gradient(X,y,omega,lamda)
     r = Norm(Gradient(X,y,omega,lamda))
     print(r)
     if( r < epsilon*r0 ):
         break     
#%%
         
     
#%%
n=0
y_star=X@omega
for i in range(y.shape[0]):
    if y_star[i]*y[i]>0:
        n+=1;
print(n)        
#%%
         
#%%

     
#%%