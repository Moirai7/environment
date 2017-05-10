#!/usr/bin/env python
import numpy as np
import h5py
 
a=np.loadtxt('data.csv',delimiter=',',skiprows=1)
print a.shape

tmpA=np.zeros((97191,7))
tmpA[:,0:7]=a[:,2:9]
normA=np.zeros(np.shape(tmpA))
print tmpA.shape
minVals=tmpA.min(0)
maxVals=tmpA.max(0)
ranges=maxVals-minVals
m=tmpA.shape[0]
normA=tmpA-np.tile(minVals,(m,1))
normA=normA/np.tile(ranges,(m,1))

#normA=normA.reshape(97191,1,7,1)
#a=a.reshape(97191,1,9,1)
b=np.zeros((97191,1))
b[:,0]=a[:,1]

#f=h5py.File("train.h5","r")
#x=f["data"]
f2=h5py.File("mine.h5","w")
f2.create_dataset("data",data=normA,dtype='float32')
#f2.create_dataset("data",data=a,dtype='float32')
f2.create_dataset("label",data=b,dtype='float32')

from sklearn.cross_validation import train_test_split
trainData,testData,trainLabel,testLabel = train_test_split(normA,b,test_size=0.10,random_state=531)
#trainData=normA[0:87000]
#trainLabel=b[0:87000]
#testData=normA[87000:]
#testLabel=b[87000:]
print testData.shape
print trainData.shape
f3=h5py.File("train.h5","w")
f3.create_dataset("data",data=trainData,dtype='float32')
f3.create_dataset("label",data=trainLabel,dtype='float32')

f4=h5py.File("test.h5","w")
f4.create_dataset("data",data=testData,dtype='float32')
f4.create_dataset("label",data=testLabel,dtype='float32')


