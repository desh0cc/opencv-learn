import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import read_image

def hsv_segmentation(image: str):
    img = read_image(image)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lowerBound = np.array([0,0,50])
    upperBound = np.array([10,120,200])

    mask = cv.inRange(imgHSV, lowerBound, upperBound)

    debug = 1

    cv.imshow('mask', mask)
    cv.waitKey(0)

def callback(input):
    pass

def segmentation_window(image: str):
    img = read_image(image)

    winname = 'hsv segment'
    cv.namedWindow(winname)

    cv.createTrackbar('lh',winname,1,180,callback)
    cv.createTrackbar('ls',winname,2,255,callback)
    cv.createTrackbar('lv',winname,3,255,callback)

    cv.createTrackbar('uh',winname,1,180,callback)
    cv.createTrackbar('us',winname,2,255,callback)
    cv.createTrackbar('uv',winname,3,255,callback)


    while True:
        if cv.waitKey(1) == ord('q'):
            break

        lh = cv.getTrackbarPos('lh',winname)
        ls = cv.getTrackbarPos('ls',winname)
        lv = cv.getTrackbarPos('lv',winname)

        uh = cv.getTrackbarPos('uh',winname)
        us = cv.getTrackbarPos('us',winname)
        uv = cv.getTrackbarPos('uv',winname)

        lowerBound = np.array([lh,ls,lv])
        upperBound = np.array([uh,us,uv])
        
        filtered = cv.inRange(img,lowerBound,upperBound)
        cv.imshow(winname,filtered)

    cv.destroyAllWindows()



if __name__ == "__main__":
    segmentation_window('el_gato.jpg')