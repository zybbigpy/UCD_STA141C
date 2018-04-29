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
y = y.reshape(y.size,1)
#random omega vector initialize
vec_omega = np.random.rand(12).reshape(12,1)
lamda = 1
eta = 0.000000000000001
epsilon = 0.00000001
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
    g = (X.transpose().dot(X.dot(omega)-y)*2/y.size + omega*lamda).A
    return g        
#%%


#%%
r0 = L2Norm(GradientVec(X_array,vec_omega,y,lamda))
# =============================================================================
# while (True):
#     vec_omega = vec_omega-eta*GradientVec(X_array,vec_omega,y,lamda)
#     r = L2Norm(GradientVec(X_array,vec_omega,y,lamda))
#     print(r)
#     if( r < epsilon*r0 ):
#         break
# =============================================================================
        
    
print(vec_omega)
#%%

#%%
TotalData = np.hstack((X_array,y))
np.random.shuffle(TotalData)
Step = int(X_array.shape[0]/4)
#train data set
TrainDataSetX = []
TrainDataSetY= []
#test data set
TestDataSetX = []
TestDataSetY = []


for i in range (4):
    a, b = i*Step, (i+1)*Step
    SubTestDataX = TotalData[a:b,:-1]
    SubTestDataY = TotalData[a:b,-1]
    SubTrainDataX = np.vstack((TotalData[0:a,:-1],TotalData[b:,:-1]))
    SubTrainDataY = np.vstack((TotalData[0:a,-1],TotalData[b:,-1]))
    TestDataSetX.append(SubTestDataX)
    TestDataSetY.append(SubTestDataY)
    TrainDataSetX.append(SubTrainDataX)
    TrainDataSetY.append(SubTrainDataY)
#%%









