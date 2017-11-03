import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
x = tf.placeholder(tf.float32,shape=(),name='x')

w1 = tf.constant(-1.0)
b = tf.constant(-0.5)

A = tf.Variable(0,name='solution')

A = x*w1 > b

sess = tf.InteractiveSession()

print(sess.run(A, feed_dict={x: 1.0}))
print(sess.run(A, feed_dict={x: 0.0}))