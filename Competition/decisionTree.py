import sklearn
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import robust_scale
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    train = pd.read_csv('train_data2.csv')
    test = pd.read_csv('test_data.csv')
    valid = pd.read_csv('train_data2.csv')
    sub = pd.read_csv('sample_submission.csv')
    sub['target'] = 0
    sub.to_csv('sub0.csv', index=False)  ## 0.58
    feature_names = [x for x in train.columns if x not in [ 'target']]
    target = train['target']
    train = train[feature_names]
    validateTarget = valid['target']
    valid = valid[feature_names]
    train = robust_scale(train)
    valid = robust_scale(valid)
    maxi = 0
    besti = 0
    maxs = 0
    bests = 0
    X_train, X_valid, y_train, y_valid = train_test_split(train, target, train_size=0.9,
                                                          stratify=target,
                                                          random_state=2017, shuffle=True)
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import svm
    from sklearn.svm import SVC
    ## set up model
    for i in range(300, 301):
        clf2 = SVC()
        clf2.fit(X_train, y_train)
        print(clf2.score(X_train, y_train))
        print(clf2.score(X_valid, y_valid))
        # print(clf2.score(valid, validateTarget))
        from sklearn.metrics import confusion_matrix

        # print(confusion_matrix(validateTarget, clf2.predict(valid)), i)
        print(confusion_matrix(y_train, clf2.predict(X_train)), i)
        print(confusion_matrix(y_valid, clf2.predict(X_valid)), i)
        if clf2.score(valid, validateTarget) > maxi:
            maxi = clf2.score(valid, validateTarget)
            bests = i
        print(maxi, bests)

        # print(confusion_matrix(validateTarget, clf2.predict(valid)))
        pred2 = clf2.predict(test[feature_names])
        ## make submission
        sub = pd.read_csv('sample_submission.csv')
        sub['target'] = pred2
        sub['target'] = sub['target'].astype(int)
        sub.to_csv('sub2'+str(i)+'.csv', index=False)
