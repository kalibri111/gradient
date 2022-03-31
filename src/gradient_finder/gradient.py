import cv2 as cv
import numpy as np
from src.geometry.geometry import Rectangle, Point
from src.algorithm.algorithm import zero_submatrix
from src.monotone_finder.monotone import is_monotone, from_particular_sum


def draw_rectangle(img: np.ndarray, rectangle: Rectangle, color, thickness) -> np.ndarray:
    return cv.rectangle(
        img,
        (rectangle.up_left.y, rectangle.up_left.x),
        (rectangle.down_right.y, rectangle.down_right.x),
        color=color,
        thickness=thickness
    )


def gradient_positions(img: np.ndarray, grayscaled: np.ndarray, without_monotone=True):
    found = None
    if without_monotone:
        def with_img(area):
            return is_monotone(area, grayscaled) and \
                   from_particular_sum(area, img) / (area.down_right.x - area.up_left.x) * (
                               area.down_right.y - area.up_left.y) <= 1

        found = zero_submatrix(img, with_img)
    else:
        found = zero_submatrix(img)

    return found
