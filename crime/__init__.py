import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import robust_scale
from sklearn import preprocessing


class Batch:
    def __init__(self, data_set, type: str, length: int):
        self.data_set = data_set.values
        self.i = 0
        self.type = type
        self.length = length

    def next(self, num):
        m = self.i
        self.i += num
        if self.i > self.length:
            m = 0
            self.i = 0
            fin = self.length
        else:
            fin = m + num
        x_val = self.data_set[m:fin][:, 1:71]
        if self.type == 'train':
            ys = self.data_set[m:fin][:, 71]
            return x_val, to_onehot(ys)
        return x_val


if __name__ == '__main__':
    def predictOnTest(a: int):
        test.i = 0
        pred = sess.run(tf.argmax(y, 1), feed_dict={x: test.next(11430)})
        print(pred.shape)
        sub = pd.read_csv('sample_submission.csv')
        sub['Criminal'] = pred
        sub['Criminal'] = sub['Criminal'].astype(int)
        sub.to_csv('deepNeuraldeeeper' + str(a) + '.csv', index=False)


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
            save_path = saver.save(sess, "/tmp/model.ckpt")
            print("Model saved in file: %s" % save_path)
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


    def read_test():
        test_csv = pd.read_csv('criminal_test.csv')
        feature_names = [x for x in test_csv.columns if x not in ['PERID', 'Criminal']]
        test_csv[feature_names] = robust_scale(test_csv[feature_names])
        return Batch(test_csv, 'test', 11430)


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
            l = int(l)
            outlabels[i, l] = 1
        return outlabels


    def read_train():
        train_csv = pd.read_csv('criminal_train - Copy.csv')
        feature_names = [x for x in train_csv.columns if x not in ['PERID', 'Criminal']]
        train_csv[feature_names] = robust_scale(train_csv[feature_names])
        train_ = train_csv.iloc[:60000]
        dev_ = train_csv.iloc[60000:]
        return Batch(train_, 'train', 60000), Batch(dev_, 'train', 17469)


    train, dev = read_train()
    test = read_test()
    # # normalizing
    # # min_max_scaler = preprocessing.StandardScaler()
    # # train = min_max_scaler.fit_transform(train)
    # # testing_data = min_max_scaler.fit_transform(testing_data)

    # These will be inputs
    ## Input pixels, flattened
    x = tf.placeholder("float", [None, 70])
    ## Known labels
    y_ = tf.placeholder("float", [None, 2])


    def hiddenLayer(previous_layer, num_previous_hidden, num_hidden: int):
        W = tf.Variable(tf.truncated_normal([num_previous_hidden, num_hidden],
                                            stddev=2. / np.math.sqrt(num_previous_hidden), mean=1))
        b = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
        h = tf.nn.relu(tf.matmul(previous_layer, W) + b)
        return h


    num_of_input = 70
    num_of_hidden_units_in_layer_1 = 800
    num_of_hidden_units_in_layer_2 = 80
    num_of_hidden_units_in_layer_3 = 100
    num_of_hidden_units_in_layer_4 = 150
    num_of_hidden_units_in_layer_5 = 200
    num_of_hidden_units_in_layer_6 = 200
    num_of_hidden_units_in_layer_7 = 100
    num_of_hidden_units_in_layer_8 = 150
    num_of_output_classes = 2
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


    y = output_layer(layer2, num_of_hidden_units_in_layer_2, num_of_output_classes)
    ### End model specification, begin training code

    # Climb on cross-entropy
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y + 1e-50))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    learning_rate = tf.placeholder(dtype=tf.float32, shape=())
    # How we train
    train_step = tf.train.RMSPropOptimizer(learning_rate, centered=True).minimize(cross_entropy)
    i = 0
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    l = 0.001
    saver.restore(sess, "/tmp/model.ckpt")
    while i != -10:
        i += 1
        x_values, y_real = train.next(100)
        train_step.run(feed_dict={x: x_values, y_: y_real, learning_rate: l})
        if i % 500 == 1:
            xx, yyy_ = dev.next(17469)
            x_values, y_real = train.next(60000)
            train_accuracy = accuracy.eval(feed_dict={x: x_values, y_: y_real, learning_rate: l})
            print('step %d, training accuracy %g' % (i, train_accuracy * 100))
            print('dev accuracy %g' % (
                    accuracy.eval(feed_dict={x: xx, y_: yyy_}) * 100))
        if i % 100000 == 0:
            l /= 2
            print(l)
            DoIpredictOnTest(i)
            i = stop(i)
