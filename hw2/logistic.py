# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 15:29:57 2018

@author: MiaoWangqian
"""

import pickle as pk
import sklearn as sk
import numpy as np
from multiprocessing import Pool, freeze_support
import time

### sigmoid function
def h(x):
    return 1/(1+np.exp(x))

### gradient after vectorization by python
def Gradient(l,omega):
    #l is a tuple
    X,y = l
    #a = np.array(sparse.csr_matrix.todense(X@omega))
    #b = np.array(sparse.csr_matrix.todense(y))
    #g =  X.T@sparse.csr_matrix(-b*h(b*a))+ lamda*omega
    a = X@omega
    b = y
    g= X.T@(-b*h(b*a))
    return g
     
### L2 Norm
def Norm(vec):
    return np.sqrt(vec.T@vec)[0,0]

### SLice dataset for Pool.map
def Slice(X,y):
        step1 = int(X.shape[0]/4)
        step2 = int(y.shape[0]/4)
        args = []
        ### cut the data into four equal size part
        for i in range(4):
                A = X[step1*i:step1*(i+1),:]
                B = y[step2*i:step2*(i+1)]
                args.append((A,B))
        return args 

### load data and return X,y
def load_data(filename):
    print("load file:", filename)
    fin = open(filename, "rb")
    data = pk.load(fin, encoding='iso-8859-1')
    X = np.array(data[0])
    Y = data[1].reshape(data[1].shape[0], 1)
    test_X = np.array(data[2])
    test_Y = data[3].reshape(data[3].shape[0], 1)
    print("Dataset Normalized ")
    max_X = np.array([np.linalg.norm(X[:, i]) for i in range(data[0].shape[1])]).reshape(1, data[0].shape[1])
    max_Y = np.linalg.norm(Y)
    X /= max_X
    Y /=max_Y
    test_X /=max_X
    test_Y /=max_Y
    return X, test_X, Y, test_Y


def main():
        ### change encoding for Windows 
        X_train,X_test,y_train,y_test=load_data("data_files.pl")
        size = int(X_train.shape[1])
        omega = np.random.randn(size).reshape(size,1)
        lamda = 1    
        
        ### train the logistic model
        r0 = Norm(Gradient((X_train,y_train),omega)+lamda*omega)
        ### hyperparameter
        epsilon = 0.001
        eta = 0.01
        ### start timer
        start = time.time()
        lst = Slice(X_train,y_train)
        pool = Pool(4)
        while (True):
                ##################################
                ### parallelization by pool.map###
                ##################################
                grads = 0
                ### return a list of omega from four dataset
                ### starmap for multiAgrs (new for python3.4) 
                grads = pool.starmap(Gradient,[(l,omega) for l in lst])
                ### update parameters 
                tmp = np.sum(np.array(grads))+omega*lamda
                r = Norm(tmp)
                print(r)
                if( r < epsilon*r0 ):
                        break  
                omega = omega - eta*(tmp)
                ##################################
                ### parallelization by pool.map###
                ##################################
                ### stop interation condition
                
   
        ### test the logistic model
        n1 = 0
        y_star1 = X_test@omega
        for i in range(y_test.shape[0]):
            if (y_star1[i]*y_test[i] > 0):
                n1 += 1
    
        ### print accuracy for testdata
        print("test acc:", n1/y_test.shape[0]) 
        ### print total time
        print("total time:",time.time()-start)
        
if __name__=="__main__":
        freeze_support()
        main()
