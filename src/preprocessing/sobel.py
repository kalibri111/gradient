import cv2 as cv
import numpy as np

_SCALE = 1
_DELTA = 0
_DDEPTH = cv.CV_16S


def greyscale_sobel(path: str) -> np.ndarray:
    """
    Convert image to gray-scaled, filter by Sobel convolution
    :param path: path to image
    :return: 2-D array
    """
    image = cv.imread(path, cv.IMREAD_GRAYSCALE)

    grad_x = cv.Sobel(image, _DDEPTH, 1, 0, ksize=3, scale=_SCALE, delta=_DELTA, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(image, _DDEPTH, 0, 1, ksize=3, scale=_SCALE, delta=_DELTA, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad
