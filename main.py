print "hello world"

import numpy as np
import cv2 as cv
from sklearn import datasets
import matplotlib.pyplot as plt

print np.array([[1,2],[3,4]])



import preprocessing
images, gt = preprocessing.readFiles("middleburry-third", 3)
print len(images), len(gt)

disp = preprocessing.computeDisparities(images)
print len(disp)

for d in disp:
    plt.imshow(d, 'gray')
    plt.show()
