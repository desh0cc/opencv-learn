import cv2 as cv
import os

FOLDER = "images"

def read_image(path: str, show: bool = True):
    img = cv.imread(os.path.join(FOLDER, path))

    if img is not None:
        cv.imshow('img', img) if show else None
        cv.waitKey(0)
        return img
    else:
        print(f"Error: Could not read image from {path}")

def write_image(image, name: str):
    path = os.path.join(FOLDER, name)
    cv.imwrite(path, image)

if __name__ == "__main__":
    img = read_image("test_img.jpg", show=False)
    write_image(img, "output.png")