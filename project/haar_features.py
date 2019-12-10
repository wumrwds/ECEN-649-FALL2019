from typing import Iterable

import numpy as np


def enum(**enums):
    return type('Enum', (), enums)


FeatureType = enum(TWO_HORIZONTAL=(2, 1), TWO_VERTICAL=(1, 2), THREE_HORIZONTAL=(3, 1), THREE_VERTICAL=(1, 3),
                   FOUR=(2, 2))


class Feature:
    def __init__(self, x: int, y: int, width: int, height: int, type: FeatureType):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.type = type

    def __call__(self, integral_image: np.ndarray) -> float:
        try:
            return np.sum(np.multiply(integral_image[self.coords_y, self.coords_x], self.coeffs))
        except IndexError as e:
            raise IndexError(str(e) + ' in ' + str(self))

    def __repr__(self):
        return f'{self.__class__.__name__}(x={self.x}, y={self.y}, width={self.width}, height={self.height})'


class Feature2h(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height, FeatureType.TWO_HORIZONTAL)
        hw = width // 2
        self.coords_x = [x, x + hw, x, x + hw,
                         x + hw, x + width, x + hw, x + width]
        self.coords_y = [y, y, y + height, y + height,
                         y, y, y + height, y + height]
        self.coeffs = [1, -1, -1, 1,
                       -1, 1, 1, -1]


class Feature2v(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height, FeatureType.TWO_VERTICAL)
        hh = height // 2
        self.coords_x = [x, x + width, x, x + width,
                         x, x + width, x, x + width]
        self.coords_y = [y, y, y + hh, y + hh,
                         y + hh, y + hh, y + height, y + height]
        self.coeffs = [-1, 1, 1, -1,
                       1, -1, -1, 1]


class Feature3h(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height, FeatureType.THREE_HORIZONTAL)
        tw = width // 3
        self.coords_x = [x, x + tw, x, x + tw,
                         x + tw, x + 2 * tw, x + tw, x + 2 * tw,
                         x + 2 * tw, x + width, x + 2 * tw, x + width]
        self.coords_y = [y, y, y + height, y + height,
                         y, y, y + height, y + height,
                         y, y, y + height, y + height]
        self.coeffs = [-1, 1, 1, -1,
                       1, -1, -1, 1,
                       -1, 1, 1, -1]


class Feature3v(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height, FeatureType.THREE_VERTICAL)
        th = height // 3
        self.coords_x = [x, x + width, x, x + width,
                         x, x + width, x, x + width,
                         x, x + width, x, x + width]
        self.coords_y = [y, y, y + th, y + th,
                         y + th, y + th, y + 2 * th, y + 2 * th,
                         y + 2 * th, y + 2 * th, y + height, y + height]
        self.coeffs = [-1, 1, 1, -1,
                       1, -1, -1, 1,
                       -1, 1, 1, -1]


class Feature4(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height, FeatureType.FOUR)
        hw = width // 2
        hh = height // 2
        self.coords_x = [x, x + hw, x, x + hw,  # upper row
                         x + hw, x + width, x + hw, x + width,
                         x, x + hw, x, x + hw,  # lower row
                         x + hw, x + width, x + hw, x + width]
        self.coords_y = [y, y, y + hh, y + hh,  # upper row
                         y, y, y + hh, y + hh,
                         y + hh, y + hh, y + height, y + height,  # lower row
                         y + hh, y + hh, y + height, y + height]
        self.coeffs = [1, -1, -1, 1,
                       -1, 1, 1, -1,
                       -1, 1, 1, -1,
                       1, -1, -1, 1]


def gen_features(width: int, height: int, feature_type: FeatureType) -> Iterable[Feature]:
    features = list()

    for size_x in range(feature_type[0], width + 1, feature_type[0]):
        for size_y in range(feature_type[1], height + 1, feature_type[1]):
            for x in range(width - size_x + 1):
                for y in range(height - size_y + 1):
                    if feature_type == FeatureType.TWO_HORIZONTAL:
                        features.append(Feature2h(x, y, size_x, size_y))
                    elif feature_type == FeatureType.TWO_VERTICAL:
                        features.append(Feature2v(x, y, size_x, size_y))
                    elif feature_type == FeatureType.THREE_HORIZONTAL:
                        features.append(Feature3h(x, y, size_x, size_y))
                    elif feature_type == FeatureType.THREE_VERTICAL:
                        features.append(Feature3v(x, y, size_x, size_y))
                    elif feature_type == FeatureType.FOUR:
                        features.append(Feature4(x, y, size_x, size_y))
    return features
