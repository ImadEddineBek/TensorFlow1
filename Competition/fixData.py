import pandas as pd
import numpy as np

for i in range(100000):
    v = 2996 * (1-i/1000000)
    if v == int(v):
        print((1-i/100))