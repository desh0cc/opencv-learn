import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import read_image

def detect_corner(image: str):
    img = read_image(image)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    grayImg = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    grayImg = np.float32(grayImg)


    plt.subplot(131)
    plt.imshow(img)

    block = 5
    sobel = 3
    k = 0.04

    harris = cv.cornerHarris(grayImg,block,sobel,k) # type: ignore

    plt.subplot(132)
    plt.imshow(harris)
    plt.subplot(133)
    img[harris>0.05*harris.max()] = [255,0,0]
    plt.imshow(img)

    plt.show()


if __name__ == '__main__':
    detect_corner('cornered.jpg')