import tensorflow as tf
import numpy as np

x = []
y = []
for i in range(5000):
    x.append([i, i + 1])
    y.append([i, i + 1, i + 2, i + 3])
x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)

x_ = tf.placeholder(tf.float32, shape=[None, 2])
y_ = tf.placeholder(tf.float32, shape=[None, 4])


def hiddenLayer(previous_layer, num_previous_hidden, num_hidden: int):
    W = tf.Variable(tf.truncated_normal([num_previous_hidden, num_hidden],
                                        stddev=2. / np.math.sqrt(num_previous_hidden)))
    b = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
    h = tf.nn.relu(tf.matmul(previous_layer, W) + b)
    return h


num_of_input = 2
num_of_hidden_units_in_layer_1 = 41
num_of_hidden_units_in_layer_2 = 80
num_of_hidden_units_in_layer_3 = 100
num_of_hidden_units_in_layer_4 = 150
num_of_hidden_units_in_layer_5 = 200
num_of_hidden_units_in_layer_6 = 200
num_of_hidden_units_in_layer_7 = 100
num_of_hidden_units_in_layer_8 = 150
# input layer
layer1 = hiddenLayer(x, num_of_input, num_of_hidden_units_in_layer_1)
# hidden layers
layer2 = hiddenLayer(layer1, num_of_hidden_units_in_layer_1, num_of_hidden_units_in_layer_2)
layer3 = hiddenLayer(layer2, num_of_hidden_units_in_layer_2, num_of_hidden_units_in_layer_3)

layer4 = hiddenLayer(layer3, num_of_hidden_units_in_layer_3, num_of_hidden_units_in_layer_4)


# layer5 = hiddenLayer(layer4, num_of_hidden_units_in_layer_4, num_of_hidden_units_in_layer_5)
#
# layer6 = hiddenLayer(layer5, num_of_hidden_units_in_layer_5, num_of_hidden_units_in_layer_6)
# layer7 = hiddenLayer(layer6, num_of_hidden_units_in_layer_6, num_of_hidden_units_in_layer_7)
# layer8 = hiddenLayer(layer7, num_of_hidden_units_in_layer_7, num_of_hidden_units_in_layer_8)


# output layer
def output_layer(previous_layer, num_previous_hidden, num_hidden: int):
    W_last = tf.Variable(tf.truncated_normal([num_previous_hidden, num_hidden],
                                             stddev=2. / np.math.sqrt(num_previous_hidden)))
    b_last = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
    # Define model
    return tf.matmul(previous_layer, W_last) + b_last


y__ = output_layer(layer1, num_of_hidden_units_in_layer_1, 4)
### End model specification, begin training code


# Climb on cross-entropy
cross_entropy = tf.reduce_mean(tf.square(y__ - y_))
# How we train
train_step = tf.train.RMSPropOptimizer(0.000001, centered=True).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(5000000):
    sess.run(train_step, feed_dict={x_: x, y_: y})
    print(sess.run(cross_entropy, feed_dict={x_: x, y_: y}))

x = []
y = []
for i in range(5000000, 5000050):
    x.append([i, i + 1])
    y.append([i, i + i + 1, 2 * i, 3 * i])
x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)
print(sess.run(cross_entropy, feed_dict={x_: x, y_: y}))
