import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import pandas as pd
import math

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


num_hidden = 128
num_hidden2 = 256
num_hidden3 = 128
W1 = tf.Variable(tf.truncated_normal([41, num_hidden],
                                     stddev=1. / math.sqrt(41)))
b1 = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
h1 = tf.sigmoid(tf.matmul(x, W1) + b1)



W11 = tf.Variable(tf.truncated_normal([num_hidden, num_hidden2],
                                     stddev=1. / math.sqrt(num_hidden)))
b11 = tf.Variable(tf.constant(0.1, shape=[num_hidden2]))
h11 = tf.sigmoid(tf.matmul(h1, W11) + b11)


W12 = tf.Variable(tf.truncated_normal([num_hidden2, num_hidden3],
                                     stddev=1. / math.sqrt(num_hidden2)))
b12 = tf.Variable(tf.constant(0.1, shape=[num_hidden3]))
h12 = tf.sigmoid(tf.matmul(h11, W12) + b12)

num_hidden4 = 256

W13 = tf.Variable(tf.truncated_normal([num_hidden3, num_hidden4],
                                     stddev=1. / math.sqrt(num_hidden3)))
b13 = tf.Variable(tf.constant(0.1, shape=[num_hidden4]))
h13 = tf.sigmoid(tf.matmul(h12, W13) + b13)

# Output Layer
W2 = tf.Variable(tf.truncated_normal([num_hidden3, 3],
                                     stddev=1. / math.sqrt(3)))
b2 = tf.Variable(tf.constant(0.1, shape=[3]))
y = tf.nn.softmax(tf.matmul(h12, W2) + b2)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())



maxTest = 0
maxTrain = 0
maxI = 0
maxJ = 0
stuck = 0
learning_rate = 0.0001


def maximize(learning_rate):
    if learning_rate >= 1 :
        return 1
    return learning_rate * 10


def minimize(learning_rate):
    if learning_rate <= 0.000001 :
        return 0.000001
    return learning_rate / 1.6
i = 0
v= 0
while v <78.5 and i <2000:
    sess.run(train_step, feed_dict={y_: onehot_train, x: train})

    currentTest = sess.run(accuracy, feed_dict={y_: onehot_test, x: test})
    currentTrain = sess.run(accuracy, feed_dict={y_: onehot_train, x: train})
    v = currentTest
    if currentTrain > maxTrain and currentTest > maxTest :
        maxTrain = currentTrain
        maxTest = currentTest
        print(i, "better train", currentTrain)
        maxJ = i
        print(i, "better test", currentTest)
        maxI = i
        # learning_rate = minimize(learning_rate)
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        # stuck -=1
        if currentTrain > 0.78 or currentTest > 0.78:
            with open('log.txt', mode='a')as file:
                print(b1.eval(), 'b1', file=file)
                print(b11.eval(), 'b11', file=file)
                print(b12.eval(), 'b12', file=file)
                print(b2.eval(), 'b2', file=file)
                print(W1.eval(), 'W1', file=file)
                print(W12.eval(), 'W12', file=file)
                print(W2.eval(), 'W2', file=file)
                currentTest = sess.run(accuracy, feed_dict={y_: onehot_test, x: test})
                currentTrain = sess.run(accuracy, feed_dict={y_: onehot_train, x: train})
                print(currentTest, 'currentTest', file=file)
                print(currentTrain, 'currentTrain', file=file)
    #     W_Max = W
    #     B_Max = B
    stuck += 1
    print(i)
    i+=1


print(sess.run(accuracy, feed_dict={y_: onehot_train, x: train}))
print(sess.run(accuracy, feed_dict={y_: onehot_test, x: test}))
print(maxJ,maxI,maxTest,maxTrain)
testing_data = pd.read_csv('test_data.csv')
testing_data = testing_data[feature_names].values

m = sess.run(y, feed_dict={x: testing_data})


def index(l):
    j = 0
    max = -1
    for i in range(3):
        if l[i] > max:
            max = l[i]
            j = i
    return j


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
