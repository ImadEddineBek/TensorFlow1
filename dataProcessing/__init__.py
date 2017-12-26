import pandas as pd
import tensorflow as tf
import numpy as np
import time
from sklearn.utils import shuffle

if __name__ == '__main__':
    dataSetCat = pd.read_csv('mushrooms.csv')
    for r in dataSetCat.columns:
        dataSetCat[r] = dataSetCat[r].astype('category')
    data2 = pd.get_dummies(dataSetCat)
    data2 = shuffle(data2)
    data2 = shuffle(data2)
    data2 = shuffle(data2)
    train = data2.iloc[:8000, :]
    test = data2.iloc[8000:, :]

    # These will be inputs
    ## Input pixels, flattened
    x = tf.placeholder("float", [None, 117])
    ## Known labels
    y_ = tf.placeholder("float", [None, 2])


    def hiddenLayer(previous_layer, num_previous_hidden, num_hidden: int):
        W = tf.Variable(tf.truncated_normal([num_previous_hidden, num_hidden],
                                            stddev=2. / np.math.sqrt(num_previous_hidden)))
        b = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
        h = tf.nn.relu(tf.matmul(previous_layer, W) + b)
        return h


    num_of_input = 117
    num_of_hidden_units_in_layer_1 = 128 * 1
    num_of_hidden_units_in_layer_2 = 128 * 2
    num_of_hidden_units_in_layer_3 = 128 * 4
    num_of_hidden_units_in_layer_4 = 128 * 8
    num_of_hidden_units_in_layer_5 = 128 * 8
    num_of_hidden_units_in_layer_6 = 128 * 4
    num_of_hidden_units_in_layer_7 = 128 * 2
    num_of_hidden_units_in_layer_8 = 128 * 1
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


    y = output_layer(layer8, num_of_hidden_units_in_layer_8, num_of_output_classes)
    ### End model specification, begin training code

    # Climb on cross-entropy
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y + 1e-50))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    tf.summary.scalar('cross_entropy', cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    # How we train
    train_step = tf.train.RMSPropOptimizer(0.00001, centered=True).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    train_writer = tf.summary.FileWriter('learning/train',
                                         sess.graph)
    test_writer = tf.summary.FileWriter('learning/test')
    tf.global_variables_initializer().run()
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    for i in range(300):
        data = train.sample(1000)
        real_target = data.iloc[:, 0:2].values
        real_features = data.iloc[:, 2:].values
        sess.run(train_step, feed_dict={x: real_features, y_: real_target})
        summary, train_acc = sess.run([merged, accuracy * 100], feed_dict={x: real_features, y_: real_target})
        print(train_acc)
        train_writer.add_summary(summary, i)
        data = test.sample(122)
        real_target = data.iloc[:, 0:2].values
        real_features = data.iloc[:, 2:].values
        summary, test_acc = sess.run([merged, accuracy * 100], feed_dict={x: real_features, y_: real_target})
        print(test_acc)
        test_writer.add_summary(summary, i)
        time.sleep(0.5)
    # save_path = saver.save(sess, "./model.ckpt")
