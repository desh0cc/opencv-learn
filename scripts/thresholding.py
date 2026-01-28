import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import read_image

def threshold(image: str):
    img = read_image(image)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    hist = cv.calcHist([img],[0],None,[256],[0,256])
    
    # plt.figure()
    # plt.plot(hist)
    # plt.ylabel('pixels')
    # plt.xlabel('bins')

    methods = {
        cv.THRESH_BINARY: 'Binary',
        cv.THRESH_BINARY_INV: 'Binary Inv.',
        cv.THRESH_TRUNC: 'Truncated',
        cv.THRESH_TOZERO: 'ToZero',
        cv.THRESH_TOZERO_INV: 'ToZero Inv.'
    }

    plt.figure()
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title('original')

    for i, method in enumerate(methods.items()):
        _, filtered = cv.threshold(img,181,255,method[0])
        plt.subplot(2,3,i+2)
        plt.imshow(filtered, cmap='gray')
        plt.title(method[1])

    plt.show()



if __name__ == '__main__':
    threshold('cattos.jpg')