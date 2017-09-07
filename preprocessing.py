import cv2 as cv

LEFT_IMAGE_FILENAME = "view1.png"
RIGHT_IMAGE_FILENAME = "view5.png"
DISPARITY_FILENAME = "disp1.png"

def readFiles(folderPath, subfolders):
    images = list()
    groundtruth = list()
    for folder in subfolders:
        lft = cv.imread(folderPath+'/'+folder+'/'+LEFT_IMAGE_FILENAME, cv.IMREAD_GRAYSCALE)
        rgt = cv.imread(folderPath+'/'+folder+'/'+RIGHT_IMAGE_FILENAME, cv.IMREAD_GRAYSCALE)
        dsp = cv.imread(folderPath+'/'+folder+'/'+DISPARITY_FILENAME, cv.IMREAD_GRAYSCALE)
        images.append((lft, rgt))
        groundtruth.append(dsp)
    return images, groundtruth
