import numpy as np
x = np.array([1,2,3])
print(np.exp(x))

def sigmoid(x):
    s = 1 / (1+ np.exp(-x))
    return s

print(sigmoid(x))

def sigmoid_derivative(x):
    s = 1/(1+np.exp(-x))
    ds = s*(1-s)
    return ds

print(sigmoid_derivative(x))


def image2vector(image):
    v = image.reshape((image.shape[0] * image.shape[1]*image.shape[2]),1)
    return v

def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x/x_norm

x = np.array([[0,3,4],[1,6,4]])
print("normalizeRow(x) = " + str(normalizeRows(x)))

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims=True)
    s = x_exp/x_sum
    return s

print(softmax(x))

# vectorization
import time
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i] *x2[i]
toc = time.process_time()
print("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

tic = time.process_time()
outer = np.zeros((len(x1), len(x2)))
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i]*x2[j]
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
tic = time.process_time()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j]*x1[j]
toc = time.process_time()
print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

def L1(yhat, y):
    loss = sum(abs(y-yhat))
    return loss

yhat = np.array([0.9, 0.2,0.1,0.4,0.9])
y = np.array([1,0,0,1,1])
print("l1 = " + str(L1(yhat,y)))

def L2(yhat, y):
    loss = np.dot(y-yhat, y-yhat)
    return loss

yhat = np.array([.9,.2,.1,.4,.9])
y = np.array([1,0,0,1,1])
print("l2 = " + str(L2(yhat,y)))

import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import dp_ai.seaborn.lr_utils
from dp_ai.seaborn.lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "'picture.")

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    assert (w.shape == (dim,1))
    assert (isinstance(b, float) or isinstance(b,int))
    return w, b

def propagate(w,b,X,Y):

    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(np.dot(Y, np.log(A.T)) + np.dot(np.log(1-A), (1-Y).T))/m
    dw = np.dot(X, (A-Y).T)/m
    db = np.sum(A-Y)/m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}
    return grads, cost

def optimize(w,b,X,Y,num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w,b,X,Y)

    dw = grads["dw"]
    db = grads["db"]

    w = w - learning_rate*dw
    b = b - learning_rate * db

    if i % 100 == 0:
        costs.append(cost)

    if print_cost and i % 100 == 0:
        print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100,learning_rate = 0.009, print_cost = False)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    A = sigmoid(np.dot(w.T, X), b)
    for i in range(A.shape[1]):
        if A[0][j] <= 0.5:
            A[0][j] = 0
        else:
            A[0][j] = 1
    Y_prediction = A
    assert(Y_prediction.shape == (1,m))
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5,print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w,b,X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T


# d = model(train_set_x, train_set_y, test_set_x, test_set_y,num_iterations = 2000, learning_rate = 0.005, print_cost = True)








