import pickle as cPickle
# cPickle is overloaded by pickle in python3
import numpy as np
import time
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support



def go_nn(Xtrain, ytrain, Xtest, ytest):
    correct =0
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

def main():
        start_time = time.time()
        fin = open("data_files.pl", "rb")
        data = cPickle.load(fin,  encoding='iso-8859-1')
# default encoding = 'utf-8', encounter UnicodeDecodeError in python3
        Xtrain = data[0]
        ytrain = data[1]
        Xtest = data[2]
        ytest = data[3]
        X = []
        y = []
        step1 = int(Xtest.shape[0]/4)
        step2 = int(ytest.shape[0]/4)
        for i in range(4):
                A = Xtest[step1*i:step1*(i+1),:]
                B = ytest[step2*i:step2*(i+1)]
                X.append(A)
                y.append(B)
        
        n =[(Xtrain, ytrain, X[0], y[0]), (Xtrain, ytrain, X[1], y[1]), (Xtrain, ytrain, X[2],y[2]), (Xtrain, ytrain, X[3], y[3])]
        with Pool(4) as pool:
                L = pool.starmap(go_nn, n)
        print(sum(L)/4)
        t=time.time()-start_time
        print(t)
        
if __name__=="__main__":
        freeze_support()
        main()