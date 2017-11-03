import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import pandas as pd


def to_onehot(labels, nclasses=3):
    outlabels = np.zeros((len(labels), nclasses))
    for i, l in enumerate(labels):
        outlabels[i, l] = 1
    return outlabels


f = 4


def next(f):
    f += 1
    return np.exp(-f) / (1 + np.exp(-f)), f


train = pd.read_csv('train_data.csv')

feature_names = [x for x in train.columns if x not in ['connection_id', 'target']]
target = train['target'].values
train = train[feature_names].values
onehot = to_onehot(target)

indices = np.random.permutation(train.shape[0])
valid_cnt = int(train.shape[0] * 0.1)

test_idx, training_idx = indices[:valid_cnt], indices[valid_cnt:]
test, train = train[test_idx, :], train[training_idx, :]
onehot_test, onehot_train = onehot[test_idx, :], onehot[training_idx, :]

x = tf.placeholder(tf.float32, shape=(None, 41))  # shape (plusieurs,'''41''')
y_ = tf.placeholder(tf.float32, shape=(None, 3), name='y_')

W = tf.Variable(tf.ones([41, 3]))  # shape('''41''',3 = the number of classes 2 aka true or false
B = tf.Variable(tf.ones([3]))  # shape (3) the number of classes

Y = tf.nn.softmax(tf.matmul(x, W) + B)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=Y))

train_step = tf.train.GradientDescentOptimizer(0.000001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
maxTest = 0
maxTrain = 0
maxI = 0
maxJ = 0
stuck = 0
learning_rate = 0.000001
for i in range(5000):
    # if stuck > 60:
    #     learning_rate *= 100
    #     train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    #     correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_, 1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     print("*" * 15, i, f, learning_rate)
    currentTest = sess.run(accuracy, feed_dict={y_: onehot_test, x: test})
    currentTrain = sess.run(accuracy, feed_dict={y_: onehot_train, x: train})
    if currentTest > maxTest:
        maxTest = currentTest
        print(i, "better test", maxTest, i)
        maxI = i
        learning_rate /= 10
        stuck = 0
    if currentTrain > maxTrain:
        maxTrain = currentTrain
        print(i, "better train", currentTrain, i)
        maxJ = i
        learning_rate /= 10
        stuck = 0
    sess.run(train_step, feed_dict={y_: onehot_train, x: train})
    stuck += 1

print(sess.run(accuracy, feed_dict={y_: onehot_train, x: train}))
print(sess.run(accuracy, feed_dict={y_: onehot_test, x: test}))

testing_data = pd.read_csv('test_data.csv')
testing_data = testing_data[feature_names].values

m = sess.run(Y, feed_dict={x: testing_data})


def index(l):
    print(l)
    for i in range(3):
        if l[i] > 0.8:
            return i
    print("fuck", l)


def back_to_normal(array):
    list = np.zeros(len(array), dtype=np.int)
    for i, l in enumerate(array):
        list[i] = index(l)
    return list


sub = pd.read_csv('sample_submission.csv')
sub['target'] = back_to_normal(m)
sub['target'] = sub['target'].astype(int)
sub.to_csv('sub1.csv', index=False)








# from sklearn.metrics import accuracy_score
# def multAcc(pred, dtrain):
#     label = dtrain.get_label()
#     acc = accuracy_score(label, pred)
#     return 'maccuracy', acc
