import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
x = tf.placeholder(tf.float32,shape=(),name='x')
y = tf.placeholder(tf.float32,shape=(),name='y')

w1 = tf.constant(0.5)
w2 = tf.constant(0.5)
b = tf.constant(0.75)

w3 = tf.constant(1.0)
w4 = tf.constant(-2.0)
w5 = tf.constant(1.0)
b1 = tf.constant(1.0)

And = tf.Variable(0.0,dtype=tf.float32,name='solution')

AndBool = x*w1 + y*w2 > b

And = tf.cast(AndBool , tf.float32 )

sess = tf.InteractiveSession()

XOR = w3 * x + w4 * And + w5 * y >= b1

print(sess.run(XOR, feed_dict={x: 1.0, y: 1.0}))
print(sess.run(XOR, feed_dict={x: 1.0, y: 0.0}))
print(sess.run(XOR, feed_dict={x: 0.0, y: 1.0}))
print(sess.run(XOR, feed_dict={x: 0.0, y: 0.0}))