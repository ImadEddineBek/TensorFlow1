import pandas as pd
import tensorflow as tf
import numpy as np


def fix_confusion(pred, conf):
    return pred


if __name__ == '__main__':
    def to_onehot(labels, nclasses=2):
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

        sub = pd.read_csv('titanic/gender_submission.csv')
        sub['target'] = pred
        sub['target'] = sub['target'].astype(int)
        sub.to_csv('titanic/deepNeural' + str(a) + '.csv', index=False)


    def DoIpredictOnTest(i: int):
        file = open('titanic/DoIWrite', mode='r')
        rep = file.read()
        if rep == '1':
            predictOnTest(i)
            file.close()


    def stop(i: int):
        file = open('titanic/DoIStop', mode='r')
        rep = file.read()
        if rep == '1':
            file.close()
            file = open('titanic/DoIWrite', mode='w')
            file.write('1')
            file.close()
            DoIpredictOnTest(i)
            i = -10
            file = open('titanic/DoIStop', mode='w')
            file.write('')
            file.close()
        return i


    train = pd.read_csv('titanic/train.csv')
    testing_data = pd.read_csv('titanic/test.csv')
    map = {'male': 1, 'female': 0}
    train['Sex'] = train['Sex'].map(map)
    testing_data['Sex'] = testing_data['Sex'].map(map)
    map = {'C': 1, 'S': 0, 'Q': 2}
    train['Embarked'] = train['Embarked'].map(map)
    testing_data['Embarked'] = testing_data['Embarked'].map(map)
    # extracting features and targets
    feature_names = [x for x in train.columns if x not in ['Survived', 'PassengerId']]
    target = train['Survived'].values
    train = train[feature_names].values
    testing_data = testing_data[feature_names].values
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
    x = tf.placeholder("float", [None, 7])
    ## Known labels
    y_ = tf.placeholder("float", [None, 2])


    def hiddenLayer(previous_layer, num_previous_hidden, num_hidden: int):
        W = tf.Variable(tf.truncated_normal([num_previous_hidden, num_hidden],
                                            stddev=2. / np.math.sqrt(num_previous_hidden)))
        b = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
        h = tf.nn.relu(tf.matmul(previous_layer, W) + b)
        return h


    num_of_input = 7
    num_of_hidden_units_in_layer_1 = 80
    num_of_hidden_units_in_layer_2 = 200
    num_of_hidden_units_in_layer_3 = 350
    num_of_hidden_units_in_layer_4 = 500
    num_of_hidden_units_in_layer_5 = 200
    num_of_hidden_units_in_layer_6 = 350
    num_of_hidden_units_in_layer_7 = 200
    num_of_hidden_units_in_layer_8 = 50
    num_of_output_classes = 2
    # input layer
    layer1 = hiddenLayer(x, num_of_input, num_of_hidden_units_in_layer_1)
    # hidden layers
    layer2 = hiddenLayer(layer1, num_of_hidden_units_in_layer_1, num_of_hidden_units_in_layer_2)
    layer3 = hiddenLayer(layer2, num_of_hidden_units_in_layer_2, num_of_hidden_units_in_layer_3)

    # layer4 = hiddenLayer(layer3, num_of_hidden_units_in_layer_3, num_of_hidden_units_in_layer_4)
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
        return tf.nn.softmax(tf.matmul(previous_layer, W_last) + b_last)


    y = output_layer(layer3, num_of_hidden_units_in_layer_3, num_of_output_classes)
    ### End model specification, begin training code

    # Climb on cross-entropy
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y + 1e-50))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # How we train
    train_step = tf.train.RMSPropOptimizer(0.00001, centered=True).minimize(cross_entropy)

    indices = np.random.permutation(train.shape[0])
    i = 0
    max_dev = 0
    max_train = 0
    max_dev_index = 0
    max_train_index = 0

    tf.global_variables_initializer().run()

    while i != -10:

        sess.run(train_step, feed_dict={x: train, y_: onehot_train})
        if i % 200 == 0:
            train_acc = sess.run(accuracy, feed_dict={x: train, y_: onehot_train})
            dev_acc = (sess.run(accuracy, feed_dict={x: dev, y_: onehot_dev}))

            if train_acc > max_train:
                max_train = train_acc
                max_train_index = i
                print("                                  better train", max_train, i)
            pred = sess.run(tf.argmax(y, 1), feed_dict={x: trainin})
            conf = tf.confusion_matrix(labels=target, predictions=pred, num_classes=2)
            print(sess.run(conf))
            if dev_acc > max_dev:
                max_dev = dev_acc
                max_dev_index = i
                print("                                  better dev", max_dev, i)
            print(i, train_acc, dev_acc, max_train_index, max_dev_index)

            DoIpredictOnTest(i)
            i = stop(i)
        i += 1
