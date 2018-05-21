# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 15:29:57 2018

@author: MiaoWangqian
"""
import pickle as pk # cPickle is overloaded by pickle in python3
import numpy as np
import time
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support



def go_nn(Xtrain, ytrain, Xtest, ytest):
    correct = 0
    for i in range(Xtest.shape[0]): ## For all testing instances
        nowXtest = Xtest[i,:]
        ### Find the index of nearest neighbor in training data
        dis_smallest = np.linalg.norm(Xtrain[0,:]-nowXtest) 
        idx = 0
        for j in range(1, Xtrain.shape[0]):
            dis = np.linalg.norm(nowXtest-Xtrain[j,:])
            if dis < dis_smallest:
                dis_smallest = dis
                idx = j
        ### Now idx is the index for the nearest neighbor
        ## check whether the predicted label matches the true label
        if ytest[i] == ytrain[idx]:  
            correct += 1
    acc = correct/float(Xtest.shape[0])
    return acc

def createArgs():
        """
        output: the args for pool.starmap(go_nn)
        cut the dataset into four different pieces
        """
        ### load dataset
        fin = open("data_files.pl", "rb")
        ### change encoding for Windows 
        data = pk.load(fin,  encoding='iso-8859-1')
        Xtrain = data[0]
        ytrain = data[1]
        Xtest = data[2]
        ytest = data[3]
        step1 = int(Xtest.shape[0]/4)
        step2 = int(ytest.shape[0]/4)
        args = []
        ### cut the data into four equal size part
        for i in range(4):
                A = Xtest[step1*i:step1*(i+1),:]
                B = ytest[step2*i:step2*(i+1)]
                args.append((Xtrain,ytrain,A,B))
        return args       
                
def main():
        ### timer on
        start_time = time.time()
        ### create 4 processor by Pool
        with Pool(4) as pool:
                ### return a list of the accuracy of four different part 
                ### use four different testdata on four processors to run go_nn 
                ### starmap for multiAgrs (new for python3.4)
                acc = pool.starmap(go_nn, createArgs())
        ### total accuracy is the average of four different part
        print("The acc is:",sum(acc)/4)
        t = time.time()-start_time
        ### print run time
        print("The toatl running time",t)
        
if __name__ == "__main__":
        freeze_support()
        main()