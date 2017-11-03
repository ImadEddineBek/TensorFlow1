# -----------------------------------

#
#   In this exercise you will put the finishing touches on a perceptron class
#
#   Finish writing the activate() method by using numpy.dot and adding in the thresholded
#   activation function

import numpy


class Perceptron:
    weights = [1]
    threshold = 0

    def activate(self, values):
        '''Takes in @param values, a list of numbers.
        @return the output of a threshold perceptron with
        given weights and threshold, given values as inputs.
        '''

        # YOUR CODE HERE

        # TODO: calculate the strength with which the perceptron fires
        v = numpy.sum(values * self.weights)
        # TODO: return 0 or 1 based on the threshold
        result = v > self.threshold
        return result

    def __init__(self, weights=None, threshold=None):
        if weights.any():
            self.weights = weights
        if threshold:
            self.threshold = threshold


x = numpy.array([1, 1, 1, 1])
w = numpy.array([1, 1, 1, 1])
per = Perceptron(w, 2)
print(per.activate(x))