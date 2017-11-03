import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
x = tf.placeholder(tf.float32,shape=(4,2))#shape (4,'''2''')
y_ = tf.placeholder(tf.float32, shape=(4,2), name='y_')

W = tf.Variable(tf.ones([2,2]))#shape('''2''',2 = the number of classes 2 aka true or false
B = tf.Variable(tf.ones([2]))#shape (2) the number of classes

Y = tf.nn.softmax(tf.matmul(x, W)+B)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=Y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for _ in range(50):
    sess.run(train_step, feed_dict={y_: [[0, 1], [0, 1], [0, 1], [1, 0]], x :[[1.0, 1.0],[ 1.0, 0.0],[ 0.0, 1.0],[ 0.0, 0.0]]})


correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={y_: [[0, 1], [0, 1], [0, 1], [1, 0]], x :[[1.0, 1.0],[ 1.0, 0.0],[ 0.0, 1.0],[ 0.0, 0.0]]}))