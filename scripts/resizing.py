import cv2 as cv
import matplotlib.pyplot as plt
from utils import read_image

def image_interp(image: str):
    img = read_image(image)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    h, w, _ = img.shape

    scale = 1/4

    methods = [
        (cv.INTER_AREA, "AREA"),
        (cv.INTER_LINEAR, "LINEAR"),
        (cv.INTER_NEAREST, "NEAREST"),
        (cv.INTER_CUBIC, "CUBIC"),
        (cv.INTER_LANCZOS4, "LANCZOS"),
    ]

    plt.figure(figsize=(10, 6))
    plt.subplot(2,3,1)
    plt.imshow(img)
    plt.title('ORIGINAL')

    for i, (method, name) in enumerate(methods):
        resized = cv.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=method
        )
        plt.subplot(2, 3, i + 2)
        plt.imshow(resized)
        plt.title(name)

    plt.show()


if __name__ == '__main__':
    image_interp('el_gato.jpg')