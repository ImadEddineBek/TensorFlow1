# list = list(map(int, input().split()))
# m = list[0]
# n = list[1]
# matrix = []
#
#
# def arrr(param: str):
#     arr = []
#     for s in param:
#         arr.append(s)
#     return arr
#
#
# for _ in range(m):
#     matrix.append(arrr(input()))
# perimeters = []
# def rest(ma:list, kkkk, iiii):
#     newMatrix = []
#     for ii , l in enumerate(ma):
#         if ii >= kkkk :
#             ligne = []
#             for j , v in enumerate(l):
#                 if j >= iiii:
#                     ligne.append(v)
#             newMatrix.append(ligne)
#     return newMatrix
#
#
# def column(looking:list, param:int):
#     newMatrix = []
#     for ii, l in enumerate(looking):
#         for j, v in enumerate(l):
#             if j == param:
#                 newMatrix.append(v)
#     return newMatrix
#
#
# def biggestPerimeter(matrix, k: int, i: int):
#     looking = rest(matrix,k, i)
#     for c in looking[0]:
#         if c == 'x':
#             return 0
#     for c in column(looking,0):
#         if c == 'x':
#             return 0
#     for c in looking[-1]:
#         if c == 'x':
#             return 0
#     for c in column(looking,-1):
#         if c == 'x':
#             return 0
#     return 2 * (len(looking) - 1) + 2 * (len(looking[0]) - 1)
#
# #
# for k in range(m - 1):
#     line = matrix[k]
#     for i in range(n - 1):
#         if line[i] == '.':
#             perimeters.append(biggestPerimeter(matrix, k, i))
# max = max(perimeters)
# if max == 0:
#     print('impossible')
# else:
#     print(max)
#
# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# import numpy as np
# import tensorflow as tf
# import pandas as pd
# import math
#
#
# def to_onehot(labels, nclasses=3):
#     outlabels = np.zeros((len(labels), nclasses))
#     for i, l in enumerate(labels):
#         outlabels[i, l] = 1
#     return outlabels
#
#
# f = 4
#
#
# def next(f):
#     f += 1
#     return np.exp(-f) / (1 + np.exp(-f)), f
#
#
# train = pd.read_csv('train_data.csv')

def mediane(list: list):
    if len(list) % 2 == 1:
        return list[(len(list)) // 2]
    else:
        return (list[(len(list)) // 2] + list[(len(list)) // 2 - 1]) / 2

length = int(input())
list = list(map(int, input().split()))
list.sort()
n = 0
upper = []
lower = []
if len(list) % 2 == 0:
    n = len(list) // 2 - 1
    lower = list[:len(list) // 2]
    upper = list[len(list) // 2:]
    q2 = (list[(len(list)) // 2] + list[(len(list)) // 2 - 1]) / 2
else:
    n = len(list) // 2
    q2 = list[n]
    lower = list[:len(list) // 2]
    upper = list[len(list) // 2 + 1:]
q1 = mediane(lower)
q3 = mediane(upper)
print(q1)
print(q2)
print(q3)

