import numpy as np
from util import to_float_array, open_face


def float_array_to_integral_img(img: np.ndarray) -> np.ndarray:
    integral = np.cumsum(np.cumsum(img, axis=0), axis=1)
    return np.pad(integral, (1, 1), 'constant', constant_values=(0, 0))[:-1, :-1]


def file_to_integral_img(images: list) -> list:
    ret = []
    ret.extend([float_array_to_integral_img(to_float_array(open_face(f))) for f in images])
    return ret


class IntegralImage:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.coords_x = [x, x + width, x, x + width]
        self.coords_y = [y, y, y + height, y + height]
        self.coeffs = [1, -1, -1, 1]

    def get_sum(self, integral_image: np.ndarray) -> float:
        return np.sum(np.multiply(integral_image[self.coords_y, self.coords_x], self.coeffs))
