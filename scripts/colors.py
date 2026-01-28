import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

FOLDER = 'images'

def pure_colors():
    zeros = np.zeros((100,100))
    ones = np.ones((100,100))

    blue = cv.merge((zeros,zeros,255*ones))
    green = cv.merge((zeros,255*ones,zeros))
    red = cv.merge((255*ones,zeros,zeros))

    black = cv.merge((zeros,zeros,zeros))
    white = cv.merge((ones,ones,ones))

    plt.figure()

    plt.subplot(231)
    plt.imshow(blue)
    plt.title('blue')

    plt.subplot(232)
    plt.imshow(green)
    plt.title('green')

    plt.subplot(233)
    plt.imshow(red)
    plt.title('red')

    plt.subplot(224)
    plt.imshow(white)
    plt.title('white')

    plt.subplot(223)
    plt.imshow(black)
    plt.title('black')

    plt.show()

def grayscale(path: str):
    img = cv.imread(os.path.join(FOLDER, path))
    
    if img is None:
        raise ValueError(f"Could not read image from {path}")
    
    plt.figure()
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cmap='gray')
    plt.show()

def color_channels(path: str):
    img = cv.imread(os.path.join(FOLDER, path))
    
    if img is None:
        raise ValueError(f"Could not read image from {path}")

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    r,g,b = cv.split(img)
    zeros = np.zeros_like(r)

    plt.figure()

    plt.subplot(131)
    plt.imshow(cv.merge((r, zeros, zeros)))
    plt.title('red channel')

    plt.subplot(132)
    plt.imshow(cv.merge((zeros,g,zeros)))
    plt.title('green channel')

    plt.subplot(133)
    plt.imshow(cv.merge((zeros,zeros,b)))
    plt.title('blue channel')


    plt.show()

if __name__ == '__main__':
    # pure_colors()
    # color_channels('el_gato.jpg')
    grayscale('el_gato.jpg')