import cv2 as cv
import matplotlib.pyplot as plt
from utils import read_image


def gray_histogram(image: str):
    img = read_image(image)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    plt.figure()
    plt.imshow(img, cmap='gray')

    hist = cv.calcHist([img], [0], None, [256],[0,256])

    plt.figure()
    plt.plot(hist)
    plt.xlabel('bins')
    plt.ylabel('pixels')
    plt.show()


def color_histogram(image: str):
    img = read_image(image)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(img)

    colors = ['r','g','b']

    plt.figure()
    for i, col in enumerate(colors):
        hist = cv.calcHist([img], [i], None, [256],[0,256])
        plt.plot(hist, col)

    plt.xlabel('bins')
    plt.ylabel('pixels')
    plt.show()



if __name__ == "__main__":
    gray_histogram('el_gato.jpg')
    color_histogram('el_gato.jpg')