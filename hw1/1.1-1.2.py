# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 09:13:09 2018

@author: MiaoWangqi

"""

#%%
from sklearn import datasets
from scipy import sparse
import numpy as np
import sklearn as sk
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
eta = 0.0001
epsilon = 0.001
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
    g = (X.transpose().dot(X.dot(omega)-y)*2/y.size + omega*lamda)
    return g        
#%%


#%%
r0 = L2Norm(GradientVec(X_array,vec_omega,y,lamda))

while (True):
     vec_omega = vec_omega-eta*GradientVec(X_array,vec_omega,y,lamda)
     r = L2Norm(GradientVec(X_array,vec_omega,y,lamda))
     print(r)
     if( r < epsilon*r0 ):
         break

print(vec_omega)
#%%

#%%
TotalData = np.hstack((X_array,y))
np.random.shuffle(TotalData)
Step = int(X_array.shape[0]/5)
#train data set
TrainDataSetX = []
TrainDataSetY= []
#test data set
TestDataSetX = []
TestDataSetY = []

for i in range (5):

    a, b = int(i*Step), int((i+1)*Step)
    SubTestDataX = TotalData[a:b,:-1]
    SubTestDataY = TotalData[a:b,-1]
    SubTrainDataX = np.vstack((TotalData[0:a,:-1],TotalData[b:,:-1]))
    SubTrainDataY = np.vstack((TotalData[0:a,-1:],TotalData[b:,-1:]))
    TestDataSetX.append(SubTestDataX)
    TestDataSetY.append(SubTestDataY)
    TrainDataSetX.append(SubTrainDataX)
    TrainDataSetY.append(SubTrainDataY)
#%%

#%%
def train(X_train,omega,y_train,lamda):
    r0 = L2Norm(GradientVec(X_train,omega,y_train,lamda))

    while (True):
        omega = omega-eta*GradientVec(X_train,omega,y_train,lamda)
        r = L2Norm(GradientVec(X_train,omega,y_train,lamda))
        print(r)
        if( r < epsilon*r0 ):
            break
    return omega

def MSE(X_test,omega,y_test):
    return L2Norm(X_test.dot(omega)-y_test)/y_test.size
#%%

#%%
mse=0  
omega = np.random.rand(12).reshape(12,1)
for i in range(5):
    omega=train(TrainDataSetX[i],omega,TrainDataSetY[i],lamda)
    mse+=MSE(TestDataSetX[i],omega,TestDataSetY[i])
print(mse/5)
#%%






