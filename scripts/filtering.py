import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import read_image

from images_rw import write_image

def median_filt(image: str):
    img = read_image(image)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    filter = cv.medianBlur(img,5)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.title('original')

    plt.subplot(122)
    plt.imshow(filter)
    plt.title('median filter')

    plt.show()

    write_image(filter,'denoised.png')

def gauss_kernel(size,sigma):
    kernel = cv.getGaussianKernel(size,sigma).flatten()
    kernel = np.outer(kernel.astype(np.float64), kernel.astype(np.float64))
    return kernel

def callback(input):
    print('changed')

def gaussian_filt(image: str):
    img = read_image(image)

    n = 51
    kernel = gauss_kernel(n,8)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(kernel)

    ax = fig.add_subplot(122,projection='3d')
    x = np.arange(0,n,1)
    y = np.arange(0,n,1)

    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,kernel,cmap='viridis')
    
    plt.show()

    winname = 'gauss filter'
    cv.namedWindow(winname)
    cv.createTrackbar('size',winname,50,1000,callback)
    cv.createTrackbar('sigma',winname,2,15,callback)

    while True:
        if cv.waitKey(1) == ord('q'):
            break

        sigma = cv.getTrackbarPos('sigma',winname)
        size = cv.getTrackbarPos('size',winname) 

        if size <= 0:
            size = 1
        if size % 2 == 0:
            size += 1

        kernel = gauss_kernel(size,sigma)
        filtered = cv.filter2D(img,-1,kernel)

        cv.imshow(winname,filtered)

    cv.destroyAllWindows()
    

if __name__ == "__main__":
    median_filt('noisy.png')
    gaussian_filt('cattos.jpg')