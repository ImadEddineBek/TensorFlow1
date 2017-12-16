import pandas as pd
import tensorflow as tf
import numpy as np
class Batch():
    def __init__(self,data_set):
        self.data_set = data_set
        self.i = 0
    def next(self,num):
        m = self.i
        self.i += num
        return self.data_set[m+num]

if __name__ == '__main__':
    def read_test():
        test = pd.read_csv()

