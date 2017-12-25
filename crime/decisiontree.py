import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import robust_scale
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

train = pd.read_csv('criminal_train.csv')
test = pd.read_csv('criminal_test.csv')
valid = pd.read_csv('criminal_train.csv')
sub = pd.read_csv('sample_submission.csv')
sub['target'] = 0
sub.to_csv('sub0.csv', index=False)  ## 0.58
feature_names = [x for x in train.columns if x not in ['PERID', 'Criminal']]
target = train['Criminal']
train = train[feature_names]
test = test[feature_names]
validateTarget = valid['Criminal']
valid = valid[feature_names]
train = robust_scale(train)
valid = robust_scale(valid)
test = robust_scale(test)
maxi = 0
besti = 0
maxs = 0
bests = 0
X_train, X_valid, y_train, y_valid = train_test_split(train, target, train_size=0.9,
                                                      stratify=target,
                                                      random_state=2017, shuffle=True)
for i in range(1000, 1001):
    print(i, '/' * 50)
    for k in range(1, 10):
        print(k, '*' * 10)
        clf2 = KNeighborsClassifier(n_neighbors=k,n_jobs=20)
        clf2.fit(X_train, y_train)
        print(clf2.score(X_train, y_train))
        print(clf2.score(X_valid, y_valid))
        from sklearn.metrics import confusion_matrix

        print(confusion_matrix(y_train, clf2.predict(X_train)), i)
        print(confusion_matrix(y_valid, clf2.predict(X_valid)), i)
        pred = clf2.predict(X_train)
        print(pred)
        count = 0
        for s in pred:
            if s == 1:
                count += 1
        print(count)
        pred = clf2.predict(X_valid)
        print(pred)
        count = 0
        for s in pred:
            if s == 1:
                count += 1
        print(count)
        pred = clf2.predict(test)
        print(pred)
        count = 0
        for s in pred:
            if s == 1:
                count += 1
        print(count)
        sub = pd.read_csv('sample_submission.csv')
        sub['Criminal'] = pred
        sub['Criminal'] = sub['Criminal'].astype(int)
        sub.to_csv('deepNeuraldeeeper' + str(k) + '   ' + str(i) + '.csv', index=False)
