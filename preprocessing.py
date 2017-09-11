import cv2 as cv
from glob import glob

LEFT_IMAGE_FILENAME = "view1.png"
RIGHT_IMAGE_FILENAME = "view5.png"
DISPARITY_FILENAME = "disp1.png"

WINDOW_SIZE = 7
# MAX_DISPARITY = 155
MAX_DISPARITY = 64
STEREO_BLOCK_SIZE = 9

def readFiles(folderPath, quantity):
    images = list()
    groundtruth = list()
    subfolders = glob(folderPath + "/*/")[:quantity]
    for folder in subfolders:
        lft = cv.imread(folder + LEFT_IMAGE_FILENAME, cv.IMREAD_GRAYSCALE)
        rgt = cv.imread(folder + RIGHT_IMAGE_FILENAME, cv.IMREAD_GRAYSCALE)
        dsp = cv.imread(folder + DISPARITY_FILENAME, cv.IMREAD_GRAYSCALE)
        images.append((lft, rgt))
        groundtruth.append(dsp)
    return images, groundtruth

def computeDisparities(images):
    disparities = list()
    stereoMatcher = cv.StereoBM_create(numDisparities=MAX_DISPARITY, blockSize=STEREO_BLOCK_SIZE)
    # stereoMatcher = cv.StereoSGBM_create(numDisparities=MAX_DISPARITY, blockSize=STEREO_BLOCK_SIZE)
    for (left, right) in images:
        disparity = stereoMatcher.compute(left, right)
        disparities.append(disparity)
    return disparities

def generateData(disparities, groundtruth):
