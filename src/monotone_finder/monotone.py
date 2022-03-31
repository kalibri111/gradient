from ..geometry.geometry import Rectangle, Point, value_at_point
import numpy as np


def is_monotone(area: Rectangle, image: np.ndarray) -> bool:
    """
    Check equality by grid
    :param image: 2d array to checking
    :param area: suspicious to monotone area
    :return: true if area monotone Rectangle
    """
    mid_up = Point(
        (area.up_left.x + area.down_right.x) // 2,
        area.up_left.y
    )

    mid_down = Point(
        (area.up_left.x + area.down_right.x) // 2,
        area.down_right.y
    )

    mid_left = Point(
        area.up_left.x,
        (area.up_left.y + area.down_right.y) // 2
    )

    mid_right = Point(
        area.down_right.x,
        (area.up_left.y + area.down_right.y) // 2
    )

    centre = Point(
        (area.up_left.x + area.down_right.x) // 2,
        (area.up_left.y + area.down_right.y) // 2
    )

    if value_at_point(image, mid_up) == \
            value_at_point(image, mid_down) == \
            value_at_point(image, mid_left) == \
            value_at_point(image, mid_right) == \
            value_at_point(image, centre):
        return True
    return False


def particular_sum(array: np.ndarray) -> np.ndarray:
    sums = np.zeros_like(array)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            upper = 0
            to_left = 0
            upper_left = 0
            if j > 0:
                upper = sums[i, j - 1]
            if i > 0:
                to_left = sums[i - 1, j]
            if i > 0 and j > 0:
                upper_left = sums[i - 1, j - 1]
            sums[i, j] = to_left + (upper - upper_left) + array[i, j]
    return sums


def from_particular_sum(area: Rectangle, array: np.ndarray):
    top_sum = 0
    left_sum = 0
    if area.up_left.y > 0:
        top_sum = array[area[1][0], area[0][1]]
    if area.up_left.x:
        left_sum = array[area[0][0], area[1][1]]

    top_left_sum = (
        array[area.up_left.x, area.up_left.y] if area.up_left.x > 0 and area.up_left.y > 0 else 0
    )
    return np.sum(
        (array[area.down_right.x, area.down_right.y] - top_sum) + (top_left_sum - left_sum),
    )
