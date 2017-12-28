import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    _x_ = []
    _y_ = []
    for i in range(0, 100):
        _x_.append(i)
        _y_.append(3 * i + 1)
    _y_ = _y_
    plt.scatter(_x_, _y_)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    a = tf.Variable(dtype=tf.float32, initial_value=0, name='a')
    b = tf.Variable(dtype=tf.float32, initial_value=0, name='b')
    x = tf.placeholder(dtype=tf.float32, shape=[None])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None])
    y = tf.add(tf.multiply(a, x), b)
    tf.summary.tensor_summary('y',y)
    tf.summary.scalar('a', a)
    tf.summary.scalar('b', b)

    loss = tf.reduce_sum(tf.square(y - y_))
    tf.summary.scalar('loss', loss)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.0000000001).minimize(loss)
    sess = tf.InteractiveSession()
    train_writer = tf.summary.FileWriter('learning/train',
                                         sess.graph)
    tf.global_variables_initializer().run()
    merged = tf.summary.merge_all()
    loss_tab = []
    for i in range(20000000):
        sess.run(train_step, feed_dict={x: _x_, y_: _y_})
        summary, loss_i = sess.run([merged, loss * 100], feed_dict={x: _x_, y_: _y_})
        print(loss_i, i)
        train_writer.add_summary(summary, i)

