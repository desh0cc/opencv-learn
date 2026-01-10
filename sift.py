import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
from utils import read_image

def sift_detect(image: str):
    img = read_image(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT.create()
    keypoints = sift.detect(gray,None)
    gray = cv.drawKeypoints(gray,keypoints,gray,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure()
    plt.imshow(gray)
    plt.show()


if __name__ == '__main__':
    sift_detect('el_gato.jpg')