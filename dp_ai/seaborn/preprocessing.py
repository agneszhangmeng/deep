from sklearn import preprocessing
import numpy as np
x = np.array([[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]])
x_scaled = preprocessing.scale(x)
#print (x_scaled)

x_scaled.mean(axis=0)
x_scaled.std(axis=0)