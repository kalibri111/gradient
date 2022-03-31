import numpy as np
from ..geometry.geometry import Point, Rectangle

_GRADIENT_CORNER = 60


def zero_submatrix(array: np.ndarray, area_filter=None) -> Rectangle:
    """
    Finding maximum zero-submatrix algorith got from https://e-maxx.ru/algo/maximum_zero_submatrix?ref=https://githubhelp.com#3
    filtering by area_filter
    :param array:
    :param area_filter:
    :return: zero-matrixed Rectangle
    """
    if area_filter is None:
        area_filter = lambda x: True

    lefts = -(
        np.ones(
            array.shape[1]
        ).astype(int)
    )
    max_area = 0

    rights = (
            np.ones(
                array.shape[1]
            ).astype(int) * array.shape[1]
    )
    tops = -(
        np.ones(
            array.shape[1]
        ).astype(int)
    )

    to_return = None

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i][j] > _GRADIENT_CORNER:
                tops[j] = i

        buffer = []
        for j in range(array.shape[1]):
            while len(buffer) and tops[buffer[-1]] <= tops[j]:
                buffer.pop()

            if buffer:
                lefts[j] = buffer[-1]
            buffer.append(j)

        buffer = []
        for j in reversed(range(array.shape[1])):
            while len(buffer) and tops[buffer[-1]] <= tops[j]:
                buffer.pop()

            if buffer:
                rights[j] = buffer[-1]
            buffer.append(j)

        for j in range(array.shape[1]):
            null_area = Rectangle(
                Point(tops[j] + 1, lefts[j] + 1),
                Point(i, rights[j] - 1)
            )

            square = (null_area.down_right.x - null_area.up_left.x) * (null_area.down_right.y - null_area.up_left.y)

            if square > 0:
                if square > max_area and not area_filter(null_area):
                    to_return = null_area
                    max_area = square

    print('Detected gradient:')
    print(f'Up left: {to_return.up_left.x}, {to_return.up_left.y}')
    print(f'Down right: {to_return.down_right.x}, {to_return.down_right.y}')
    return to_return
