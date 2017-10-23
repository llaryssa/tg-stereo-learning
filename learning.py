from __future__ import division

from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

SPLIT_RATIO = .75
CLASSIFIERS = {
    'SVC': svm.SVC(kernel='rbf', verbose=True),
    'LinearSVC': svm.LinearSVC(),
    'knn3': KNeighborsClassifier(n_neighbors=3),
    'knn5': KNeighborsClassifier(n_neighbors=5),
    'knn7': KNeighborsClassifier(n_neighbors=7)
}


def train(data, labels, method='LinearSVC'):
    print "training..."
    classifier = CLASSIFIERS[method]
    classifier.fit(data, labels)
    return classifier

def test(model, data, labels):
    print "testing..."
    prediction = model.predict(data)
    return metrics.classification_report(labels, prediction)

def splitData(data, labels):
    sz = int(len(data)*SPLIT_RATIO)
    training_data = data[:sz]
    training_labels = labels[:sz]
    testing_data = data[sz:]
    testing_labels = labels[sz:]
    return training_data, training_labels, testing_data, testing_labels

