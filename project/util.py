import numpy as np
from PIL import Image, ImageOps


def to_float_array(img: Image.Image) -> np.ndarray:
    return np.array(img).astype(np.float32) / 255.


def to_image(values: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(values * 255.))


def gamma(values: np.ndarray, coeff: float = 2.2) -> np.ndarray:
    return values ** (1. / coeff)


def gleam(values: np.ndarray) -> np.ndarray:
    return np.sum(gamma(values), axis=2) / values.shape[2]


def open_face(path: str, resize: bool = True, window_size: int = 24) -> Image.Image:
    crop_top = 50

    img = Image.open(path)
    img = to_image(gamma(to_float_array(img)[crop_top:, :]))
    min_size = np.min(img.size)
    img = ImageOps.fit(img, (min_size, min_size), Image.ANTIALIAS)
    if resize:
        img = img.resize((window_size, window_size), Image.ANTIALIAS)
    return img.convert('L')
