# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     fig = plt.figure()
#     a = fig.add_subplot(1, 3, 1)
#     img = np.array([[0, 0, 0, 10, 10, 10],
#                     [0, 0, 0, 10, 10, 10],
#                     [0, 0, 0, 10, 10, 10],
#                     [0, 0, 0, 10, 10, 10],
#                     [0, 0, 0, 10, 10, 10],
#                     [0, 0, 0, 10, 10, 10]])
#     filter = np.array([[1, 0, -1],
#                        [1, 0, -1],
#                        [1, 0, -1]])
#
#     plt.imshow(img, cmap='gray')
#     a.set_title('image')
#     a = fig.add_subplot(1, 3, 2)
#     plt.imshow(filter, cmap='gray')
#     a.set_title('filter')
#     from scipy import signal
#
#     convolved = signal.convolve2d(img, filter,mode='valid')
#     print(np.absolute(convolved))
#
#     a = fig.add_subplot(1, 3, 3)
#     plt.imshow(np.absolute(convolved), cmap='gray')
#     a.set_title('convolved')
#     plt.show()













import tensorflow as tf
if __name__ == '__main__':
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




















