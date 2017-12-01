import pandas as pd
import numpy as np

train = pd.read_csv('train_data_fav - Copy (2).csv')
target = train['target']
t0 = train[target == 0]
t1 = train[target == 1]
t2 = train[target == 2]
t0 = t0.values
t1 = t1.values
t2 = t2.values
newT0 = []
count = 0


def equa(t, tuple):
    for i in range(len(t) - 2):
        if t[i + 1] != tuple[i + 1]:
            return False
    return True


def exists(t2, tuple):
    for t in t2:
        if equa(t, tuple):
            return True
    return False


fav = []
for i in range(len(t0)):
    fav.append(t0[i % len(t0)])
    fav.append(t1[i % len(t1)])
    fav.append(t2[i % len(t2)])

pd.DataFrame(np.array(fav)).to_csv('fav22.csv', index=False)
