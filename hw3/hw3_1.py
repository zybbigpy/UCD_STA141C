# -*- coding: utf-8 -*-
"""
Created on Mon May 28 14:21:43 2018

@author: MiaoWangqian
"""


import pickle as pk # cPickle is overloaded by pickle in python3
import numpy as np

### these helper functions are inspired by Justin Wang's code
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
               return np.zeros((1,122))
        else:
               return np.mean(cluster,axis=0)
    



def main():
        ### load data
        X = pk.load(open("C:/Users/MiaoWangqian/Desktop/hw3/data_dense.pl","rb"),encoding='latin1')
        X = np.array(X)
        ### create cluster
        clusters = [[] for i in range(10)]
        centers = [np.random.randn(1,X.shape[1]) for i in range(10)]
        ### k-means iteration
        for iter in range(40):
                totalDist = 0
                for i in range(X.shape[0]):
                        a =  X[i,:].reshape(1,122)
                        c, dist = assign_cluster(a,centers)
                        totalDist += dist
                        clusters[c].append(a)
                print("obejective function:",totalDist)
                centers =[cluster_mean(np.array(i)) for i in clusters]  


if __name__ == "__main__":
        main()