# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:55:41 2018

@author: MiaoWangqian
"""
#%%
import pickle as pk # cPickle is overloaded by pickle in python3
import numpy as np
from scipy import sparse

X = pk.load(open("C:/Users/MiaoWangqian/Desktop/hw3/data_sparse_E2006.pl","rb"),encoding='latin1')


def distance(a,b):
        dis = np.sqrt((a-b)@(a-b).T)
        return dis[0,0]

def assign_cluster(a,centers):
        dists = np.array([distance(a,x) for x in centers])
        a = np.argmin(dists)
        dist = dists[a] 
        return np.argmin(dists),dist*dist

def cluster_mean(cluster):
        if cluster.shape[0]==0:
               return np.zeros((1,))
        else:
               return np.mean(cluster,axis=0)
#%%
    



#%%
clusters = [[] for i in range(10)]
centers = [X[i,:] for i in range(10)]
for iter in range(40):
        totalDist = 0
        for i in range(X.shape[0]):
                a =  X[i,:]
                c, dist= assign_cluster(a,centers)
                totalDist+=dist
                clusters[c].append(i)
        print(totalDist)
        centers =[sparse.csr_matrix(np.sum(X[i,:],axis=0)/len(i)) for i in clusters]  
#%%
