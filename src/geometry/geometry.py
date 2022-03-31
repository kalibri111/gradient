from collections import namedtuple
import numpy as np

Point = namedtuple('Point', ('x', 'y'))
Rectangle = namedtuple('Rectangle', ('up_left', 'down_right'))


def value_at_point(array: np.ndarray, point: Point):
    return array[point.x, point.y]
