print "hello world"

import numpy as np
import cv2 as cv
from sklearn import datasets
import matplotlib.pyplot as plt

print np.array([[1,2],[3,4]])



import preprocessing
images, gt = preprocessing.readFiles("middleburry-third", ["Baby1", "Baby3"])
print len(images), len(gt)
