import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)


def initialise_weight_variable(shape, in_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=in_name)

def initialise_bias_variable(shape, in_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=in_name)

def convolution_2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

X = tf.placeholder(tf.float32, [None, 784], name='X') # These are the inputs
Y_ = tf.placeholder(tf.float32, [None, 10], name='Y_') # This is the labels, i.e the target results

# First convolutional layer
W_conv_1 = initialise_weight_variable([5, 5, 1, 32], 'W_conv_1') # First two 5s relate to the 5x5 patch used in the convolution, 1 is the number of channels (here is greyscale) & 32 is the number of output features
b_conv_1 = initialise_bias_variable([32], 'b_conv_1')

X_image = tf.reshape(X, [-1, 28, 28, 1], name='X_image')

convolution_1 = tf.nn.relu(convolution_2d(X_image, W_conv_1) + b_conv_1, name='convolution_1')
pool_1 = max_pool_2x2(convolution_1) # image now 14x14
#pool_1.name('pool_1')

# Second Convolutional layer

W_conv_2 = initialise_weight_variable([5, 5, 32, 64], 'W_conv_2') #  There are now 32 input channels and 64 outputs
b_conv_2 = initialise_bias_variable([64], 'b_conv_2')

convolution_2 = tf.nn.relu(convolution_2d(pool_1, W_conv_2) + b_conv_2, name='convolution_2')
pool_2 = max_pool_2x2(convolution_2) # image now 7x7
#pool_2.name() = 'pool_2'

# Densely Connected Layer 

W_fully_connected_1 = initialise_weight_variable([7 * 7 * 64, 1024], 'W_fully_connected_1')
b_fully_connected_1 = initialise_bias_variable([1024], 'b_fully_connected_1')

pool_2_flat = tf.reshape(pool_2, [-1, 7 * 7 * 64], 'pool_2_flat')
fully_connected_1 = tf.nn.relu(tf.matmul(pool_2_flat, W_fully_connected_1) + b_fully_connected_1, name='fully_connected_1')

# Drop out
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
fully_connected_1_drop = tf.nn.dropout(fully_connected_1, keep_prob, name='fully_connected_1_drop')

# Readout layer

W_fully_connected_2 = initialise_weight_variable([1024, 10], 'W_fully_connected_2')
b_fully_connected_2 = initialise_bias_variable([10], 'b_fully_connected_2')

Y_conv = tf.matmul(fully_connected_1_drop, W_fully_connected_2) + b_fully_connected_2
#Y_conv.name() = 'Y_conv'

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y_conv))
train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(Y_conv, 1), tf.argmax(Y_, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

def floatify(img):
    data = list(img.getdata())
    new_data = []
    element = 0
    for i in range(len(data)):
        element = float((255 - data[i])/255)
        new_data.append(element)
    return new_data

iterations = 5000
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(iterations):
    batch =  mnist.train.next_batch(50)
    if i % 1000 == 0:
      train_accuracy = accuracy.eval(feed_dict={X:batch[0], Y_:batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' %(i, train_accuracy))
    train_op.run(feed_dict={X: batch[0], Y_: batch[1], keep_prob: 0.5})
  print('test accuracy %g' %accuracy.eval(feed_dict={X:mnist.test.images, Y_:mnist.test.labels, keep_prob: 1.0}))

  #These must be changed to include the actual file directory when ran
  img_0 = Image.open('0.bmp')
  img_1 = Image.open('1.bmp')
  img_2 = Image.open('2.bmp')
  img_3 = Image.open('3.bmp')
  img_4 = Image.open('4.bmp')
  img_5 = Image.open('5.bmp')
  img_6 = Image.open('6.bmp')
  img_7 = Image.open('7.bmp')
  img_8 = Image.open('8.bmp')
  img_9 = Image.open('9.bmp')

  data_0 = floatify(img_0)
  data_1 = floatify(img_1)
  data_2 = floatify(img_2)
  data_3 = floatify(img_3)
  data_4 = floatify(img_4)
  data_5 = floatify(img_5)
  data_6 = floatify(img_6)
  data_7 = floatify(img_7)
  data_8 = floatify(img_8)
  data_9 = floatify(img_9)

  image_feed = [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9]
  null_array = np.zeros([10, 10])
  null_array.astype(float)

  print("0 : %s" %np.argmax(sess.run(Y_conv, feed_dict={X: image_feed, Y_: null_array, keep_prob: 1.0})[0]))
  print("1 : %s" %np.argmax(sess.run(Y_conv, feed_dict={X: image_feed, Y_: null_array, keep_prob: 1.0})[1]))
  print("2 : %s" %np.argmax(sess.run(Y_conv, feed_dict={X: image_feed, Y_: null_array, keep_prob: 1.0})[2]))
  print("3 : %s" %np.argmax(sess.run(Y_conv, feed_dict={X: image_feed, Y_: null_array, keep_prob: 1.0})[3]))
  print("4 : %s" %np.argmax(sess.run(Y_conv, feed_dict={X: image_feed, Y_: null_array, keep_prob: 1.0})[4]))
  print("5 : %s" %np.argmax(sess.run(Y_conv, feed_dict={X: image_feed, Y_: null_array, keep_prob: 1.0})[5]))
  print("6 : %s" %np.argmax(sess.run(Y_conv, feed_dict={X: image_feed, Y_: null_array, keep_prob: 1.0})[6]))
  print("7 : %s" %np.argmax(sess.run(Y_conv, feed_dict={X: image_feed, Y_: null_array, keep_prob: 1.0})[7]))
  print("8 : %s" %np.argmax(sess.run(Y_conv, feed_dict={X: image_feed, Y_: null_array, keep_prob: 1.0})[8]))
  print("9 : %s" %np.argmax(sess.run(Y_conv, feed_dict={X: image_feed, Y_: null_array, keep_prob: 1.0})[9]))
