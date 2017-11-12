# # -----------------------------------
#
# #
# #   In this exercise you will put the finishing touches on a perceptron class
# #
# #   Finish writing the activate() method by using numpy.dot and adding in the thresholded
# #   activation function
#
# import numpy
#
#
# class Perceptron:
#     weights = [1]
#     threshold = 0
#
#     def activate(self, values):
#         '''Takes in @param values, a list of numbers.
#         @return the output of a threshold perceptron with
#         given weights and threshold, given values as inputs.
#         '''
#
#         # YOUR CODE HERE
#
#         # TODO: calculate the strength with which the perceptron fires
#         v = numpy.sum(values * self.weights)
#         # TODO: return 0 or 1 based on the threshold
#         result = v > self.threshold
#         return result
#
#     def __init__(self, weights=None, threshold=None):
#         if weights.any():
#             self.weights = weights
#         if threshold:
#             self.threshold = threshold
#
#
# # x = numpy.array([1, 1, 1, 1])
# # w = numpy.array([1, 1, 1, 1])
# # per = Perceptron(w, 2)
# # print(per.activate(x))
# n = int(input())
# m = int(input())
# list = []
# for i in range(n):
#     list.append(int(input()))
# index = -1
# for l,i in enumerate(list) :
#     if i == m :
#         index = l
# print(index)
# n = int(input())
#
# liste = []
# for _ in range(n):
#     st = list(input().split())
#     liste.append(st)
# params = list(input().split())
# key = int(params[0])
# if params[1] == 'false':
#     reversed = False
# else:
#     reversed = True
# type = params[2]
#
#
# def toInt(liste):
#     newListe = []
#     for i in range(len(liste)):
#         st = []
#         for l in range(len(liste[i])):
#             st.append(int(liste[i][l]))
#         newListe.append(st)
#     return newListe
#
#
# def listeATrie(liste, key):
#     newList = []
#     for i in range(len(liste)):
#         newList.append(liste[i][key-1])
#     return newList
#
#
# def indexesSorted(newLi):
#     cop = newLi.copy()
#     ind = [b[0] for b in sorted(enumerate(cop),key=lambda i:i[1])]
#     return ind
# real = liste
# if type == 'numeric':
#     liste = toInt(liste)
# newLi =  listeATrie(liste, key)
# index = indexesSorted(newLi)
# if reversed :
#     i = n -1
#     while i >= 0:
#         print(' '.join(real[index[i]]))
#         i-=1
# else :
#     for i in range(len(liste)):
#         print(' '.join(real[index[i]]))




























