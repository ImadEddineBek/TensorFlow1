import numpy
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
def plot(tab):

    images = numpy.array(mnist.train.images)[tab]
    labels = numpy.array(mnist.train.lables)[tab]
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    for i in range(1,7):
        a = fig.add_subplot(2, 3, i)
        print(images[0])
        plt.imshow(np.reshape(images[i],newshape=(28,28)), cmap='gray')
        a.set_title(str(labels[i]))

    plt.show()