# -*- coding: utf-8 -*-
import numpy as np
import pylab as pl

train = np.random.rand(15,3)
obj = train[:,0] + train[:,1]*2 + train[:,0]*3 

maxStep = 2000
step = 0.1
                      
N = train.shape[1]
w1 = np.random.rand(N,2*N-1)
w2 = np.random.rand(2*N-1,1)
dw1 = np.zeros((N,2*N-1))
dw2 = np.zeros((2*N-1,1))
e = np.array([[1]])

flagCnt = 0
eList = [0]
cnt = 0
testCnt = 0
while e>0.001:
    e = np.array([[0]])
    dw1 = np.zeros((N,2*N-1))
    dw2 = np.zeros((2*N-1,1))
    
    for i in range(train.shape[0]):
        x = train[i,:]
        x = x.reshape(1,N)
        y = obj[i]
        
        #正向传播
        v =1/(1+np.exp(-1*np.dot( x,w1))) 
        v =v.T
        output = 1/(1+np.exp(-1*np.dot(v.T,w2)))
        e = e+0.5*(y-output)**2
        
        #反向传播
        dw2 += (y-output)*output*(1-output)*v
        dw1 += (y-output)*output*x.T*((1-output)*w2*v*(1-v)).T
               
    eList.append(float(e))
    
    if abs(e-eList[-2])/e < 0.001:
        flagCnt += 1
    else:
        flagCnt = 0
    
    if flagCnt > 5:
       w1 = np.random.rand(N,2*N-1)
       w2 = np.random.rand(2*N-1,1)
       testCnt +=1
    else:
        w2 += step*dw2
        w1 += step*dw1
    
    cnt += 1
    if cnt>maxStep:
        print("CAN NOT CONVERGENCE")
        break
    
pl.plot(range(len(eList)),eList)
pl.show()
