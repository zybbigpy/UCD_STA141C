import numpy as np
from sklearn import model_selection
from sklearn import datasets
from scipy import sparse


#some of the methods are learned from the machine learning written by Prof.Zhihua Zhou from my home university.


def loadDataSet():
    srcDataMat, srcLabelMat=datasets.load_svmlight_file("D:/UCD_STA141C/hw1/news20.binary.bz2")
    srcdatam,srcdatan=np.shape(srcDataMat)
    srcLabelMat=np.mat(srcLabelMat)
    for i in range(srcdatam):
        if(srcLabelMat[0,i]==-1):
            srcLabelMat[0,i]=0
    '''Split the source data into two part:train and test'''
    shuffleRes = model_selection.StratifiedShuffleSplit(1, 0.2, 0.8, 0)
    for trainIndex, testIndex in shuffleRes.split(srcDataMat, srcLabelMat.transpose()):
        trainDataMat, testDataMat = srcDataMat[trainIndex], srcDataMat[testIndex]
        trainLabelMat, testLabelMat = srcLabelMat.transpose()[trainIndex], srcLabelMat.transpose()[testIndex]

    '''Apply the logisticRegression to the train set to obtain the weights matrix'''
    weights=logisticRegression(trainDataMat,trainLabelMat)
    '''Apply the weight matrix to the test set to obtain the recall rate and precision rate'''
    binaryClassifierToTest(testDataMat, testLabelMat.transpose(), weights)

def logisticRegression(dataMatIn, classLabelsMatIn):
    datam, datan = np.shape(dataMatIn)
    featureZero =np.ones((datam,1))
    '''Add the first column as feature Zero and it equates to (1,1,……,1)'''
    modifiedDataMatIn = sparse.hstack([featureZero,dataMatIn])
    return gradDescent(modifiedDataMatIn, classLabelsMatIn)


def sigmoidFunc(parax):
    return np.longfloat(1.0 / (1.0 + np.exp(-parax)))


def gradDescent(dataMatIn, classLabelsMatIn):
    labelMat = classLabelsMatIn
    dataMatrix = dataMatIn
    datam, datan = np.shape(dataMatrix)
    '''Step length'''
    alpha = 0.001
    '''Iterative times'''
    maxIterCycles = 3000
    '''Initial weight matrix'''
    weights = np.ones((datan, 1))
    for k in range(maxIterCycles):
        hx = sigmoidFunc(dataMatrix * weights)
        error = (hx - labelMat)
        weights = weights - alpha * dataMatrix.transpose() * error
    return weights

def binaryClassifierToTest(dataMatIn, classLabelMatIn, weights):
    datam,datan=np.shape(dataMatIn)
    featureZero=np.ones((datam,1))
    '''Add the first column as feature Zero and it equates to (1,1,……,1)'''
    modifiedDataMatIn=sparse.hstack([featureZero,dataMatIn])
    modifiedDataMatIn=modifiedDataMatIn.tocsr()
    '''Structure three variables used to calculate precision rate and recall rate'''
    resMatrix=np.zeros((2,2))

    '''The process below is used to estimate the test sample and obtain the result'''
    for i in range(datam):
        dataTestSample=modifiedDataMatIn[i,:]
        '''The process below is used to judge whether a test sample should be classified into Postive Class or not'''
        if(sigmoidFunc(dataTestSample*weights)>0.5):
            if(classLabelMatIn[0,i]>0.5):
                resMatrix[0,0]=resMatrix[0,0]+1
            else:
                resMatrix[1,0]=resMatrix[1,0]+1
        else:
            if(classLabelMatIn[0,i]>0.5):
                resMatrix[0,1]=resMatrix[0,1]+1
            else:
                resMatrix[1,1]=resMatrix[1,1]+1
    '''Calculate the two ratio: precision rate AND recall rate'''
    print("-->Precision Rate:", resMatrix[0,0]/(resMatrix[0,0]+resMatrix[1,0]))

if __name__ == '__main__':
    loadDataSet()