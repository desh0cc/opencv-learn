import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import read_image


def callback(input):
    pass

def canny_edge(image: str):
    img = read_image(image)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    height,width,_ = img.shape
    scale = 1/1

    height,width = int(height*scale), int(width*scale)
    img = cv.resize(img,(height,width),interpolation=cv.INTER_LINEAR)

    winname = 'canny'
    cv.namedWindow(winname)
    cv.createTrackbar('mint',winname,10,255,callback)
    cv.createTrackbar('maxt',winname,10,255,callback)

    while True:
        if cv.waitKey(1) == ord('q'):
            break

        mint = cv.getTrackbarPos('mint', winname)
        maxt = cv.getTrackbarPos('maxt', winname)
        filtered = cv.Canny(img,mint,maxt)
        cv.imshow(winname,filtered)
    

    cv.destroyAllWindows()
    

    



if __name__ == '__main__':
    canny_edge('cattos.jpg')