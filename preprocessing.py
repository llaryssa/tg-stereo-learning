from __future__ import division

import cv2 as cv
import numpy as np
from glob import glob

LEFT_IMAGE_FILENAME = "view1.png"
RIGHT_IMAGE_FILENAME = "view5.png"
DISPARITY_FILENAME = "disp1.png"

DEFAULT_FOLDER_NAME = "middleburry-third"
DEFAULT_QUANTITY = 21
DEFAULT_DATA_FILE = "data"
DEFAULT_LABELS_FILE = "labels"

WINDOW_SIZE = 3
HALF_WINDOW_SIZE = int(WINDOW_SIZE/2)
MAX_DISPARITY = 64
MIN_DISPARITY = 0
STEREO_BLOCK_SIZE = 5

def readFiles(folderPath, quantity):
    images = list()
    groundtruth = list()
    subfolders = glob(folderPath + "/*/")[:quantity]
    for folder in subfolders:
        print "reading ", folder

        lft = cv.imread(folder + LEFT_IMAGE_FILENAME, cv.IMREAD_GRAYSCALE)
        rgt = cv.imread(folder + RIGHT_IMAGE_FILENAME, cv.IMREAD_GRAYSCALE)
        images.append((lft, rgt))

        # divided by 3 because is a third of the original size
        dsp = cv.imread(folder + DISPARITY_FILENAME, cv.IMREAD_GRAYSCALE)
        groundtruth.append(dsp/3)
    return images, groundtruth

def computeDisparities(images):
    disparities = list()
    # stereoMatcher = cv.StereoBM_create(numDisparities=MAX_DISPARITY, blockSize=STEREO_BLOCK_SIZE)
    # stereoMatcher = cv.StereoSGBM_create(numDisparities=MAX_DISPARITY, blockSize=STEREO_BLOCK_SIZE, minDisparity=MIN_DISPARITY)
    stereoMatcher = cv.StereoSGBM_create(
        minDisparity = MIN_DISPARITY,
        numDisparities = MAX_DISPARITY,
        blockSize = STEREO_BLOCK_SIZE,
        P1 = 8*3*WINDOW_SIZE**2,
        P2 = 32*3*WINDOW_SIZE**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    for (left, right) in images:
        disparity = np.int_(stereoMatcher.compute(left, right).astype(np.float32) / 16.0)
        disparities.append(disparity)
    return disparities

def generateData(disparities, groundtruth):
    data = list()
    labels = list()
    for img_idx in range(len(disparities)):
        disp = disparities[img_idx]
        gt = groundtruth[img_idx]
        (sx, sy) = disp.shape
        for i in range(sx - WINDOW_SIZE):
            for j in range(sy - WINDOW_SIZE):
                winDisp = np.int_(disp[i:(i+WINDOW_SIZE), j:(j+WINDOW_SIZE)])
                winGt = np.int_(gt[i:(i+WINDOW_SIZE), j:(j+WINDOW_SIZE)])

                # normalizing by subtracting the center pixel
                winDisp = winDisp - winDisp[HALF_WINDOW_SIZE, HALF_WINDOW_SIZE]
                winGt = winGt - winGt[HALF_WINDOW_SIZE, HALF_WINDOW_SIZE]

                label = compareDisparities(winDisp, winGt)

                data.append(winDisp.reshape([1,winDisp.size]))
                labels.append(label)
    return data, labels

def compareDisparities(win1, win2):
    s1 = win1.sum()
    s2 = win2.sum()
    return s1==s2
    # return abs(s1-s2) <= 1

def process(folder=DEFAULT_FOLDER_NAME,
            quantity=DEFAULT_QUANTITY,
            plot=False,
            outputData=DEFAULT_DATA_FILE,
            outputLabels=DEFAULT_LABELS_FILE):
    images, groundtruth = readFiles(folder, quantity)
    disparities = computeDisparities(images)
    data, labels = generateData(disparities, groundtruth)

    positives = [x for x in labels if x]
    print "ratio true labels: ", len(positives) / len(data)


    zeros = [x for x in data if not x.any()]
    print "positives: ", len(positives)
    print "zeros: ", len(zeros)

    if (plot):
        import matplotlib.pyplot as plt
        for (d, g) in zip(disparities, groundtruth):
            fig = plt.figure()

            sub1 = fig.add_subplot(1, 2, 1)
            plt.imshow(d)
            sub1.set_title('estimate')
            plt.colorbar()

            sub2 = fig.add_subplot(1, 2, 2)
            plt.imshow(g)
            sub2.set_title('ground truth')
            plt.colorbar()

            plt.show()

    # np.save(outputData, data)
    # np.save(outputLabels, labels)
    return data, labels