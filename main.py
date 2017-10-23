from __future__ import division

import preprocessing
import learning

import time

# # data, labels = preprocessing.process(quantity=1, plot=True)
# data, labels = preprocessing.process(quantity=1)
# training_data, training_labels, testing_data, testing_labels = learning.splitData(data, labels)
# model = learning.train(training_data, training_labels)
# metrics = learning.test(model, testing_data, testing_labels)
# print metrics

time0 = time.time()

data, labels = preprocessing.process(quantity=21)
true = [i for i in labels if i]
print "true labels ratio: ", len(true), "/", len(labels), "=", len(true)/len(labels)
training_data, training_labels, testing_data, testing_labels = learning.splitData(data, labels)
print "training dataset: ", len(training_labels), " samples"
print "testing dataset: ", len(testing_labels), " samples"

time1 = time.time()
print "TIME ELAPSED GENERATING DATA: ", time1 - time0, "\n\n"

classifiers = ['knn3', 'knn5', 'knn7', 'LinearSVC', 'SVC']
# classifiers = ['SVC']
for algorithm in classifiers:
    print algorithm.upper()
    timei = time.time()

    model = learning.train(training_data, training_labels, method=algorithm)
    timei1 = time.time()
    print "TIME ELAPSED TRAINING: ", timei1 - timei

    metrics = learning.test(model, testing_data, testing_labels)
    timei2 = time.time()
    print "TIME ELAPSED TESTING: ", timei2 - timei1

    print algorithm, '\n', metrics, '\n\n'