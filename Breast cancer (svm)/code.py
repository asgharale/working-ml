import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import pickle

df = datasets.load_breast_cancer()

classes = ['malignant', 'benign']
x = df.data
y = df.target



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
best_clf = 0
best_knn = 0
for _ in range(100):
    clf = svm.SVC(kernel="linear")
    clf.fit(x_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(x_train, y_train)

    clf_predict = clf.predict(x_test)
    knn_predict = knn.predict(x_test)

    clf_acc = metrics.accuracy_score(y_test, clf_predict)
    knn_acc = metrics.accuracy_score(y_test, knn_predict)

    if clf_acc > best_clf:
        best_clf = clf_acc
        with open('clf_model.pkl', 'wb') as f:
            pickle.dump(clf ,f)
    if knn_acc > best_knn:
        best_knn = knn_acc
        with open('knn_model.pkl', 'wb') as f:
            pickle.dump(knn, f)
with open('clf_model.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)

clf_predict = clf.predict(x_test)
knn_predict = knn.predict(x_test)

clf_acc = metrics.accuracy_score(y_test, clf_predict)
knn_acc = metrics.accuracy_score(y_test, knn_predict)

