import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from utils import read_image

def gradient_image(image: str):
    img = read_image(image)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    laplace = cv.Laplacian(img,cv.CV_64F,ksize=21)

    kx,ky = cv.getDerivKernels(1,0,3)
    print(ky@kx.T)


    plt.figure()
    plt.subplot(221)
    plt.imshow(img, cmap='gray')
    
    plt.subplot(222)
    plt.imshow(laplace, cmap='gray')

    sobelX = cv.Sobel(img,cv.CV_64F,1,0,ksize=21)
    sobelY = cv.Sobel(img,cv.CV_64F,0,1,ksize=21)

    plt.subplot(223)
    plt.imshow(sobelX, cmap='gray')
    plt.title('sobelX')
    
    plt.subplot(224)
    plt.imshow(sobelY, cmap='gray')
    plt.title('sobelY')

    plt.show()




if __name__ == "__main__":
    gradient_image('cattos.jpg')