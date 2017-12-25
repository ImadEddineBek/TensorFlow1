import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
import gym
with tf.device('/gpu:0'):
    class Batch:
        def __init__(self, data_set, type: str, length: int):
            self.data_set = data_set.values
            self.i = 0
            self.type = type
            self.length = length

        def next(self, num):
            m = self.i
            self.i += num
            print(self.i)
            if self.i > self.length:
                self.i = 0
                fin = self.length
            else:
                fin = m + num
            x_val = self.data_set[m:fin][:, 1:4]
            imgs = []
            for path in self.data_set[m:fin][:, 4]:
                if self.type == 'test':
                    imgs.append(read_image_test(path))
                else:
                    imgs.append(read_image_train(path))
            x_imgs = np.array(imgs)
            if self.type == 'train':
                return x_val, x_imgs, to_onehot(self.data_set[m:fin][:, 5])
            return x_val, x_imgs


    if __name__ == '__main__':
        def predictOnTest(a: int):
            print('' * 15 + str(a) + '*' * 15)
            file = open('hey' + str(a) + '.csv', mode='a')
            for i in range(1239):
                xxxxxx, xxxxxim = test.next(10)
                pred = sess.run(tf.argmax(y_conv, 1), feed_dict={x_img: xxxxxim, x_val: xxxxxx, keep_prob: 0.5})
                for p in pred:
                    file.write(str(p) + "\n")
            file.close()


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
            test_csv = pd.read_csv('test.csv')
            return Batch(test_csv, 'test', 12386)


        def to_onehot(labels, nclasses=14):
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


        def read_train():
            train_csv = pd.read_csv('train.csv')
            train_ = train_csv.iloc[:15000]
            dev_ = train_csv.iloc[15000:]
            return Batch(train_, 'train', 15000), Batch(dev_, 'train', 3576)


        def read_image_train(path):
            data = np.array(Image.open('train_/' + path).getdata(), dtype=np.ubyte)

            try:
                data = data.reshape(1024, 1024, 1)
                return data
            except ValueError:
                try:
                    return data[:, 0].reshape(1024, 1024, 1)
                except ValueError:
                    print('train*', path)

                    initial = tf.truncated_normal([1024, 1024, 1], stddev=0.1)
                    return tf.Variable(initial)


        def read_image_test(path):
            data = np.array(Image.open('test_/' + path).getdata(), dtype=np.ubyte)
            try:
                data = data.reshape(1024, 1024, 1)
                return data
            except ValueError:
                try:
                    return data[:, 0].reshape(1024, 1024, 1)
                except ValueError:
                    print('test*', path)
                    initial = tf.truncated_normal([1024, 1024, 1], stddev=0.1)
                    return tf.Variable(initial)


        train, dev = read_train()
        test = read_test()

        # variables
        x_val = tf.placeholder("float", [None, 3])
        x_img = tf.placeholder(tf.float32, shape=[None, 1024, 1024, 1])
        y_ = tf.placeholder("float", [None, 14])


        def hiddenLayer(previous_layer, num_previous_hidden, num_hidden: int):
            W = tf.Variable(tf.truncated_normal([num_previous_hidden, num_hidden],
                                                stddev=2. / np.math.sqrt(num_previous_hidden)))
            b = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
            h = tf.nn.relu(tf.matmul(previous_layer, W) + b)
            return h


        with tf.device('/cpu:0'):

            num_of_input = 3
            num_of_hidden_units_in_layer_1 = 41
            num_of_hidden_units_in_layer_2 = 80
            num_of_hidden_units_in_layer_3 = 100
            num_of_output_classes = 14
            # input layer
            layer1 = hiddenLayer(x_val, num_of_input, num_of_hidden_units_in_layer_1)
            # hidden layers
            layer2 = hiddenLayer(layer1, num_of_hidden_units_in_layer_1, num_of_hidden_units_in_layer_2)
            layer3 = hiddenLayer(layer2, num_of_hidden_units_in_layer_2, num_of_hidden_units_in_layer_3)


            def output_layer(previous_layer, num_previous_hidden, num_hidden: int):
                W_last = tf.Variable(tf.truncated_normal([num_previous_hidden, num_hidden],
                                                         stddev=2. / np.math.sqrt(num_previous_hidden)))
                b_last = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
                # Define model
                return tf.nn.softmax(tf.matmul(previous_layer, W_last) + b_last)


            y = output_layer(layer3, num_of_hidden_units_in_layer_3, num_of_output_classes)


        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)


        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)


        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')


        W_conv1 = weight_variable([9, 9, 1, 16])
        b_conv1 = bias_variable([16])

        h_conv1 = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 16, 32])
        b_conv2 = bias_variable([32])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_conv3 = weight_variable([5, 5, 32, 64])
        b_conv3 = bias_variable([64])

        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        W_conv4 = weight_variable([5, 5, 64, 128])
        b_conv4 = bias_variable([128])

        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        h_pool4 = max_pool_2x2(h_conv4)

        W_conv5 = weight_variable([5, 5, 128, 64])
        b_conv5 = bias_variable([64])

        h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
        h_pool5 = max_pool_2x2(h_conv5)

        W_conv6 = weight_variable([5, 5, 64, 128])
        b_conv6 = bias_variable([128])

        h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)
        h_pool6 = max_pool_2x2(h_conv6)

        W_conv7 = weight_variable([5, 5, 128, 64])
        b_conv7 = bias_variable([64])

        h_conv7 = tf.nn.relu(conv2d(h_pool6, W_conv7) + b_conv7)
        h_pool7 = max_pool_2x2(h_conv7)

        W_fc1 = weight_variable([8 * 8 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool7, [-1, 8 * 8 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 14])
        # b_fc2 = bias_variable([14])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + y
        i = 0
        learning_rate = 0.00001
        with tf.device('/cpu:0'):
            # Climb on cross-entropy
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv + 1e-50))
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # How we train
            train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.InteractiveSession(config=config)

        tf.global_variables_initializer().run()
        saver = tf.train.Saver()

        while i != -10:
            i += 1
            x_values, x_images, y_real = train.next(16)
            train_step.run(feed_dict={x_img: x_images, x_val: x_values, y_: y_real, keep_prob: 1})
            if i % 5 == 1:
                xx, xxx, yyy_ = dev.next(16)
                train_accuracy = accuracy.eval(feed_dict={x_img: x_images, x_val: x_values, y_: y_real, keep_prob: 1})
                print('step %d, training accuracy %g' % (i, train_accuracy * 100))
                print('dev accuracy %g' % (
                        accuracy.eval(feed_dict={x_img: xxx, x_val: xx, y_: yyy_, keep_prob: 1}) * 100))
                with tf.device('/cpu:0'):
                    print(sess.run(tf.confusion_matrix(labels=tf.argmax(y_, 1), predictions=tf.argmax(y_conv, 1),
                                                       num_classes=14),
                                   feed_dict={x_img: x_images,
                                              x_val: x_values,
                                              y_: y_real,
                                              keep_prob: 1}))
                    print(sess.run(tf.confusion_matrix(labels=tf.argmax(y_, 1), predictions=tf.argmax(y_conv, 1),
                                                       num_classes=14),
                                   feed_dict={x_img: xxx,
                                              x_val: xx,
                                              y_: yyy_,
                                              keep_prob: 1}))
            DoIpredictOnTest(i)
            if i % 1000 == 0:
                d = []
                t = []
                for i in range(50):
                    x_values, x_images, y_real = train.next(15)
                    t.append(accuracy.eval(
                        feed_dict={x_img: x_images, x_val: x_values, y_: y_real, keep_prob: 1}) * 100)
                    xx, xxx, yyy_ = dev.next(15)
                    d.append(accuracy.eval(feed_dict={x_img: xxx, x_val: xx, y_: yyy_, keep_prob: 1}) * 100)
                print("                           ", np.array(d).sum() / 50)
                print("                           ", np.array(t).sum() / 50)
            i = stop(i)
