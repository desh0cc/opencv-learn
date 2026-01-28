import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import read_image


def hough_line(image: str):
    img = read_image(image)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(img, (7,7), 3)
    filtered = cv.Canny(img,130,180)

    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(img)
    plt.subplot(1,4,2)
    plt.imshow(blurred)
    plt.subplot(1,4,3)
    plt.imshow(filtered)

    dist = 1
    angle = np.pi/180
    threshold = 150
    lines = cv.HoughLines(filtered,dist,angle,threshold)

    k = 3000

    for line in lines:
        rho,theta = line[0] # type: ignore
        dhat = np.array([[np.cos(theta)],[np.sin(theta)]])
        lhat =  np.array([[-np.sin(theta)],[np.cos(theta)]])

        d = rho*dhat

        p1 = d + k*lhat
        p2 = d - k*lhat

        p1 = p1.astype(int)
        p2 = p2.astype(int)

        cv.line(img,(p1[0][0],p1[1][0]), (p2[0][0],p2[1][0]),(255,255,255),2)



    plt.subplot(144)
    plt.imshow(img)

    plt.show()



if __name__ == '__main__':
    hough_line('cattos.jpg')