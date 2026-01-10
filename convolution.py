import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from utils import read_image

def image_conv(image: str):
    img = read_image(image)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    n = 30
    kernel = np.ones((n,n), dtype=np.float32)/(n*n)
    filter = cv.filter2D(img, -1, kernel)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('orig')

    plt.subplot(1,2,2)
    plt.imshow(filter)
    plt.title('convoluted')

    plt.show()

def callback(input):
    pass

def avg_filter(image: str):
    img = read_image(image)

    winName = 'avg filter'
    cv.namedWindow(winName)
    cv.createTrackbar('n',winName,1,100,callback)

    h,w,_ = img.shape
    scale = 1/2

    height = int(h*scale)
    width = int(w*scale)

    # img = cv.resize(img,(width,height))

    while True:
        if cv.waitKey(1) == ord('q'):
            break
        
        n = cv.getTrackbarPos('n',winName)
        filtered = cv.blur(img,(n,n))
        cv.imshow(winName,filtered)

    cv.destroyAllWindows()

if __name__ == '__main__':
    # image_conv('el_gato.jpg')
    avg_filter('el_gato.jpg')