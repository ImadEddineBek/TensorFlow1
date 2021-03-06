import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, shape=(), name='x')
    w1 = tf.constant(-1.0,name='weight')
    b = tf.constant(-0.5,name='theta')

    A = x*w1 > b

    sess = tf.InteractiveSession()

    print(sess.run(A, feed_dict={x: 1.0}))
    print(sess.run(A, feed_dict={x: 0.0}))

writer = tf.summary.FileWriter('logs', sess.graph)
writer.close()