# -*- coding: utf-8 -*-
import numpy as np


N = 3
w1 = np.random.rand(N,2*N-1)
w2 = np.random.rand(2*N-1,1)

input = np.array([[1,2,3]])
v =1/(1+np.exp(-1*np.dot( input,w1)))  
output = 1/(1+np.exp(-1*np.dot(v,w2)))  

