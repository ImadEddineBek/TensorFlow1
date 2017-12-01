import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing


def fix_confusion(pred,conf):
    print(pred)
    print(type(pred))
    print(conf)
    print(type(conf))
    print(type(conf[0]))
    print(sess.run(conf))
    print(sess.run(conf[0][1]))
    return pred


if __name__ == '__main__':
    def to_onehot(labels, nclasses=3):
        '''
        Convert labels to "one-hot" format.

        >>> a = [0,1,2,3]
        >>> to_onehot(a,5)
        array([[ 1.,  0.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  0.,  0.],
               [ 0.,  0.,  0.,  1.,  0.]])
        '''
        outlabels = np.zeros((len(labels), nclasses))
        for i, l in enumerate(labels):
            outlabels[i, l] = 1
        return outlabels


    def predictOnTest(a: int):
        pred = sess.run(y, feed_dict={x: testing_data})

        sub = pd.read_csv('sample_submission.csv')
        sub['target'] = pred
        sub['target'] = sub['target'].astype(int)
        sub.to_csv('deepNeural' + str(a) + '.csv', index=False)


    def DoIpredictOnTest(i: int):
        file = open('DoIWrite', mode='r')
        rep = file.read()
        if rep == '1':
            predictOnTest(i)
            file.close()


    def stop(i: int):
        file = open('DoIStop', mode='r')
        rep = file.read()
        if rep == '1':
            file.close()
            file = open('DoIWrite', mode='w')
            file.write('1')
            file.close()
            DoIpredictOnTest(i)
            i = -10
            file = open('DoIStop', mode='w')
            file.write('')
            file.close()
        return i


    train = pd.read_csv('train_data2.csv')
    testing_data = pd.read_csv('test_data.csv')

    # extracting features and targets
    feature_names = [x for x in train.columns if x not in ['target']]
    target = train['target']
    train = train[feature_names]
    testing_data = testing_data[feature_names]
    # normalizing
    min_max_scaler = preprocessing.StandardScaler()
    train = min_max_scaler.fit_transform(train)
    testing_data = min_max_scaler.fit_transform(testing_data)

    onehot = to_onehot(target)

    # Split data into training and validation
    indices = np.random.permutation(train.shape[0])
    valid_cnt = int(train.shape[0] * 0.1)
    dev_idx, training_idx = indices[:valid_cnt], indices[valid_cnt:]
    trainin = train
    dev, train = train[dev_idx, :], train[training_idx, :]
    onehot_dev, onehot_train = onehot[dev_idx, :], onehot[training_idx, :]

    sess = tf.InteractiveSession()

    # These will be inputs
    ## Input pixels, flattened
    x = tf.placeholder("float", [None, 41])
    ## Known labels
    y_ = tf.placeholder("float", [None, 3])


    def hiddenLayer(previous_layer, num_previous_hidden, num_hidden: int):
        W = tf.Variable(tf.truncated_normal([num_previous_hidden, num_hidden],
                                            stddev=2. / np.math.sqrt(num_previous_hidden)))
        b = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
        h = tf.nn.relu(tf.matmul(previous_layer, W) + b)
        return h


    num_of_input = 41
    num_of_hidden_units_in_layer_1 = 41
    num_of_hidden_units_in_layer_2 = 80
    num_of_hidden_units_in_layer_3 = 100
    num_of_hidden_units_in_layer_4 = 150
    num_of_hidden_units_in_layer_5 = 200
    num_of_hidden_units_in_layer_6 = 200
    num_of_hidden_units_in_layer_7 = 100
    num_of_hidden_units_in_layer_8 = 150
    num_of_output_classes = 3
    # input layer
    layer1 = hiddenLayer(x, num_of_input, num_of_hidden_units_in_layer_1)
    # hidden layers
    layer2 = hiddenLayer(layer1, num_of_hidden_units_in_layer_1, num_of_hidden_units_in_layer_2)
    layer3 = hiddenLayer(layer2, num_of_hidden_units_in_layer_2, num_of_hidden_units_in_layer_3)

    layer4 = hiddenLayer(layer3, num_of_hidden_units_in_layer_3, num_of_hidden_units_in_layer_4)
    layer5 = hiddenLayer(layer4, num_of_hidden_units_in_layer_4, num_of_hidden_units_in_layer_5)


    layer6 = hiddenLayer(layer5, num_of_hidden_units_in_layer_5, num_of_hidden_units_in_layer_6)
    layer7 = hiddenLayer(layer6, num_of_hidden_units_in_layer_6, num_of_hidden_units_in_layer_7)
    layer8 = hiddenLayer(layer7, num_of_hidden_units_in_layer_7, num_of_hidden_units_in_layer_8)
    # output layer
    def output_layer(previous_layer, num_previous_hidden, num_hidden: int):
        W_last = tf.Variable(tf.truncated_normal([num_previous_hidden, num_hidden],
                                                 stddev=2. / np.math.sqrt(num_previous_hidden)))
        b_last = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
        # Define model
        return tf.nn.softmax(tf.matmul(previous_layer, W_last) + b_last)


    y = output_layer(layer8, num_of_hidden_units_in_layer_8, num_of_output_classes)
    ### End model specification, begin training code


    # Climb on cross-entropy
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y + 1e-50))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # How we train
    train_step = tf.train.RMSPropOptimizer(0.001, centered=True).minimize(cross_entropy)

    indices = np.random.permutation(train.shape[0])
    i = 0
    max_dev = 0
    max_train = 0
    max_dev_index = 0
    max_train_index = 0
    batch = 10000
    tf.global_variables_initializer().run()
    first = 0
    last = batch
    while i != -10:

        sess.run(train_step, feed_dict={x: train, y_: onehot_train})
        if i % 20 == 0:
            train_acc = sess.run(accuracy, feed_dict={x: train, y_: onehot_train})
            dev_acc = (sess.run(accuracy, feed_dict={x: dev, y_: onehot_dev}))

            if train_acc > max_train:
                max_train = train_acc
                max_train_index = i
                print("                                  better train", max_train, i)
                pred = sess.run(tf.argmax(y, 1), feed_dict={x: trainin})
                conf = tf.confusion_matrix(labels=target, predictions=pred, num_classes=3)
                print(sess.run(conf))
            if dev_acc > max_dev:
                max_dev = dev_acc
                max_dev_index = i
                print("                                  better dev", max_dev, i)
                pred = sess.run(tf.argmax(y, 1), feed_dict={x: trainin})
                conf = tf.confusion_matrix(labels=target, predictions=pred, num_classes=3)
                pred_confusion = fix_confusion(pred,conf)
                print(sess.run(conf))
            print(i, train_acc, dev_acc, max_train_index, max_dev_index)

            DoIpredictOnTest(i)
            i = stop(i)
        i += 1


    def predictOnTest(a: int):
        pred = sess.run(y, feed_dict={x: testing_data})

        sub = pd.read_csv('sample_submission.csv')
        sub['target'] = pred
        sub['target'] = sub['target'].astype(int)
        sub.to_csv('deepNeuraldeeeper' + str(a) + '.csv', index=False)
