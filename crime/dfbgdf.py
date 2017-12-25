import pandas as pd
train = pd.read_csv('criminal_train.csv')

feature_names = [x for x in train.columns if x not in ['PERID', 'Criminal']]
target = train['Criminal']
train = train[feature_names]
t0 = train[target == 0]
t2 = train[target == 1]
t0 = t0.values
t2 = t2.values
newT0 = []
count = 0


def equa(t, tuple):
    for i in range(len(t)-1):
        if t[i+1] != tuple[i+1]:
            return False
    return True


def exists(t2, tuple):
    for t in t2:
        if equa(t, tuple):
            return True
    return False


count = 0
for tuple in t2:
    if not exists(t0, tuple):
        count += 1
        print(count)
print(count)