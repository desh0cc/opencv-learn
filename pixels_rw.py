import cv2 as cv
import matplotlib.pyplot as plt
import os

from images_rw import write_image

IMAGE_DIR = 'images'

def readnwrite_single(path: str):
    image = os.path.join(IMAGE_DIR, path)
    img = cv.imread(image)

    if img is None:
        raise ValueError(f"Could not read image from {image}")
    
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    eyeArea = imgRGB[50, 28]
    imgRGB[50, 28] = [255, 255, 255]

    debug = 1

    plt.figure()
    plt.imshow(imgRGB)
    plt.show()

def readnwrite_region(path: str):
    image = os.path.join(IMAGE_DIR, path)
    img = cv.imread(image)

    if img is None:
        raise ValueError(f"Could not read image from {image}")

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    earRegion = imgRGB[40:100,135:200]

    debug = 1

    dy = 100-40
    dx = 200-135

    startX = 300
    startY = 250

    imgRGB[startY:startY+dy, startX:startX+dx] = earRegion

    plt.figure()
    plt.imshow(imgRGB)
    plt.show()

    return imgRGB

if __name__ == "__main__":
    # readnwrite_single('test_img.jpg')
    new_img = readnwrite_region('el_gato.jpg')
    write_image(new_img, 'new_el_gato.jpg')