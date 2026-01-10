import cv2 as cv
import os
from cv2.typing import MatLike

def read_image(path: str) -> MatLike:
    image = os.path.join('images', path)
    img = cv.imread(image)

    if img is None: 
        raise ValueError(f"Could not read image from {image} :c")
    return img