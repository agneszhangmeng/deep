from __future__ import absolute_import, division, print_function
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import theano
import tensorflow as tf
import keras
from keras.layers import Dense
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import numpy as np


# linear regression
import tflearn

# Regression data
X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]
Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]

# Linear Regression graph
input_ = tflearn.input_data(shape=[None])
linear = tflearn.single_unit(input_)
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=0.01)
m = tflearn.DNN(regression)
m.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_epoch=False)

print("\nRegression result:")
print("Y = " + str(m.get_weights(linear.W)) +
      "*X + " + str(m.get_weights(linear.b)))

print("\nTest prediction for x = 3.2, 3.3, 3.4:")
print(m.predict([3.2, 3.3, 3.4]))



# x = tf.placeholder(tf.float32, [None,784])
# y_ = tf.placeholder(tf.float32, [None, 10])
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# y = tf.matmul(x, W) + b
#
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# sess = tf.InteractiveSession()
# tf.global_variables_initializer().run()
#
# # Train
# for _ in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: mnist.test.images,
#                                       y_: mnist.test.labels}))
#
# # tensorflow
# node1 = tf.constant(3.0, dtype=tf.float32)    #set constant, no need to initiate
# node2 = tf.constant(4.0) # also tf.float32 implicitly
# sess = tf.Session()
# print(sess.run([node1, node2]))
# node3 = tf.add(node1, node2)
# print("node3", node3)
# print("sess run node3:", sess.run(node3))
#
#
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# adder_node = a + b  # + provides a shortcut for tf.add(a, b)
# print(sess.run(adder_node, {a: 3, b: 4.5}))
# print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
#
#
#
# W = tf.Variable([.3], dtype=tf.float32)
# b = tf.Variable([-.3], dtype=tf.float32)
# x = tf.placeholder(tf.float32)
# linear_model = W * x + b
# init = tf.global_variables_initializer()
# sess.run(init)
# print(sess.run(linear_model, {x:[1,2,3,4]}))
#
#
# y = tf.placeholder(tf.float32)
# squared_deltas = tf.square(linear_model - y)
# loss = tf.reduce_sum(squared_deltas)
#
# fixW = tf.assign(W, [-1.])
# fixb = tf.assign(b, [1.])
# sess.run([fixW, fixb])
# print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
#
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# for i in range(1000):
#     sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
#
# print(sess.run([W, b]))
#
#
# # 自己编写函数
# def model_fn(features, labels, mode):
#     # Build a linear model and predict values
#     W = tf.get_variable("W", [1], dtype=tf.float64)
#     b = tf.get_variable("b", [1], dtype=tf.float64)
#     y = W * features['x'] + b
#     # Loss sub-graph
#     loss = tf.reduce_sum(tf.square(y - labels))
#     # Training sub-graph
#     global_step = tf.train.get_global_step()
#     optimizer = tf.train.GradientDescentOptimizer(0.01)
#     train = tf.group(optimizer.minimize(loss),
#                      tf.assign_add(global_step, 1))
#     # EstimatorSpec connects subgraphs we built to the
#     # appropriate functionality.
#     return tf.estimator.EstimatorSpec(
#         mode=mode,
#         predictions=y,
#         loss=loss,
#         train_op=train)
#
# estimator = tf.estimator.Estimator(model_fn=model_fn)
# # define our data sets
# x_train = np.array([1., 2., 3., 4.])
# y_train = np.array([0., -1., -2., -3.])
# x_eval = np.array([2., 5., 8., 1.])
# y_eval = np.array([-1.01, -4.1, -7, 0.])
# input_fn = tf.estimator.inputs.numpy_input_fn(
#     {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
# train_input_fn = tf.estimator.inputs.numpy_input_fn(
#     {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
# eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#     {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)
#
# # train
# estimator.train(input_fn=input_fn, steps=1000)
# # Here we evaluate how well our model did.
# train_metrics = estimator.evaluate(input_fn=train_input_fn)
# eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
# print("train metrics: %r" % train_metrics)
# print("eval metrics: %r" % eval_metrics)
#
#
#
#
# # data_dim = 16
# # timesteps = 8
# # num_classes = 10
# # batch_size = 32
# #
# # # Expected input batch shape: (batch_size, timesteps, data_dim)
# # # Note that we have to provide the full batch_input_shape since the network is stateful.
# # # the sample of index i in batch k is the follow-up for the sample i in batch k-1.
# # model = Sequential()
# # model.add(LSTM(32, return_sequences=True, stateful=True,
# #                batch_input_shape=(batch_size, timesteps, data_dim)))
# # model.add(LSTM(32, return_sequences=True, stateful=True))
# # model.add(LSTM(32, stateful=True))
# # model.add(Dense(10, activation='softmax'))
# #
# # model.compile(loss='categorical_crossentropy',
# #               optimizer='rmsprop',
# #               metrics=['accuracy'])
# #
# # # Generate dummy training data
# # x_train = np.random.random((batch_size * 10, timesteps, data_dim))
# # y_train = np.random.random((batch_size * 10, num_classes))
# #
# # # Generate dummy validation data
# # x_val = np.random.random((batch_size * 3, timesteps, data_dim))
# # y_val = np.random.random((batch_size * 3, num_classes))
# #
# # model.fit(x_train, y_train,
# #           batch_size=batch_size, epochs=5, shuffle=False,
# #           validation_data=(x_val, y_val))
#
#
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# learning_rate = 0.001
# training_steps = 10000
# batch_size = 128
# display_step = 200
#
# num_input = 28 # MNIST data input (img shape: 28*28)
# timesteps = 28 # timesteps
# num_hidden = 128 # hidden layer num of features
# num_classes = 10 # MNIST total classes (0-9 digits)
#
# X = tf.placeholder("float", [None, timesteps, num_input])
# Y = tf.placeholder("float", [None, num_classes])
#
# weights = {
#     'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
# }
# biases = {
#     'out': tf.Variable(tf.random_normal([num_classes]))
# }
# def RNN(x, weights, biases):
#
#     # Prepare data shape to match `rnn` function requirements
#     # Current data input shape: (batch_size, timesteps, n_input)
#     # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
#
#     # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
#     x = tf.unstack(x, timesteps, 1)
#
#     # Define a lstm cell with tensorflow
#     lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
#
#     # Get lstm cell output
#     outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#
#     # Linear activation, using rnn inner loop last output
#     return tf.matmul(outputs[-1], weights['out']) + biases['out']
#
# logits = RNN(X, weights, biases)
# prediction = tf.nn.softmax(logits)
#
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=logits, labels=Y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# train_op = optimizer.minimize(loss_op)
# correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     # Run the initializer
#     sess.run(init)
#     for step in range(1, training_steps+1):
#         batch_x, batch_y = mnist.train.next_batch(batch_size)
#         # Reshape data to get 28 seq of 28 elements
#         batch_x = batch_x.reshape((batch_size, timesteps, num_input))
#         # Run optimization op (backprop)
#         sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
#         if step % display_step == 0 or step == 1:
#             # Calculate batch loss and accuracy
#             loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
#                                                                  Y: batch_y})
#             print("Step " + str(step) + ", Minibatch Loss= " + \
#                   "{:.4f}".format(loss) + ", Training Accuracy= " + \
#                   "{:.3f}".format(acc))
#
#     print("Optimization Finished!")
#
#     # Calculate accuracy for 128 mnist test images
#     test_len = 128
#     test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
#     test_label = mnist.test.labels[:test_len]
#     print("Testing Accuracy:", \
#         sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
#
# # softmax regression, one-hot
#
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 使用one-hot编码
# print(mnist.train.images.shape, mnist.train.labels.shape)
# print(mnist.test.images.shape, mnist.test.labels.shape)
# print(mnist.validation.images.shape, mnist.validation.labels.shape)
#
# sess = tf.InteractiveSession()
# x = tf.placeholder(tf.float32, [None, 784])  # 构建占位符，None表示样本的数量可以是任意的
# W = tf.Variable(tf.zeros([784, 10]))  # 构建一个变量，代表权重矩阵，初始化为0
# b = tf.Variable(tf.zeros([10]))  # 构建一个变量，代表偏置，初始化为0
# y = tf.nn.softmax(tf.matmul(x, W) + b)
#
# y_ = tf.placeholder(tf.float32, [None, 10])
# # 交叉熵损失函数
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#
# tf.global_variables_initializer().run()
# for i in range(1000):  # 迭代次数1000
#     batch_xs, batch_ys = mnist.train.next_batch(100)  # 使用minibatch，一个batch大小为100
#     train_step.run({x: batch_xs, y_: batch_ys})
#
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # tf.argmax()返回的是某一维度上其数据最大所在的索引值，在这里即代表预测值和真值
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 用平均值来统计测试准确率
# print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))





