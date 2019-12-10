import numpy as np
from PIL import Image, ImageDraw


def to_float_array(img: Image.Image) -> np.ndarray:
    return np.array(img).astype(np.float32) / 255.0


def to_image(values: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(values * 255.0))


def gamma(values: np.ndarray, coeff: float = 2.2) -> np.ndarray:
    return values ** (1. / coeff)


def open_face(path: str) -> Image.Image:
    img = Image.open(path)
    img = to_image(gamma(to_float_array(img)))
    return img.convert('L')


def count_labels(face_prediction, label):
    cnt = 0
    for row in face_prediction:
        if row[0] == label:
            cnt += 1
    return cnt


def draw_rectangle(image, coordinates, fill):
    draw = ImageDraw.Draw(image)
    rect_start = (coordinates[0][0], coordinates[0][1])
    rect_end = (coordinates[1][0], coordinates[1][1])
    draw.rectangle((rect_start, rect_end), fill=fill)
    return image


def draw_feature_2h(image, feature):
    img = draw_rectangle(image, ((feature.y, feature.x), (feature.y + feature.height/2 - 1, feature.x + feature.width - 1)), "yellow")
    draw_rectangle(img, ((feature.y + feature.height/2, feature.x), (feature.y + feature.height - 1, feature.x + feature.width - 1)), "red")
    return img


def draw_feature_2v(image, feature):
    img = draw_rectangle(image, ((feature.y, feature.x), (feature.y + feature.height - 1, feature.x + feature.width/2 - 1)), "yellow")
    draw_rectangle(img, ((feature.y, feature.x + feature.width/2), (feature.y + feature.height - 1, feature.x + feature.width - 1)), "red")
    return img

def draw_feature_3h(image, feature):
    img = draw_rectangle(image, ((feature.y, feature.x), (feature.y + feature.height/3 - 1, feature.x + feature.width - 1)), "yellow")
    draw_rectangle(img, ((feature.y + feature.height/3, feature.x), (feature.y + 2*feature.height/3 - 1, feature.x + feature.width - 1)), "red")
    img = draw_rectangle(image, ((feature.y + 2*feature.height/3, feature.x), (feature.y + feature.height - 1, feature.x + feature.width - 1)), "yellow")
    return img


def draw_feature_3v(image, feature):
    img = draw_rectangle(image, ((feature.y, feature.x), (feature.y + feature.height - 1, feature.x + feature.width/3 - 1)), "yellow")
    draw_rectangle(img, ((feature.y, feature.x + feature.width/3), (feature.y + feature.height - 1, feature.x + 2*feature.width/3 - 1)), "red")
    img = draw_rectangle(image, ((feature.y, feature.x + 2*feature.width/3), (feature.y + feature.height - 1, feature.x + feature.width - 1)), "yellow")
    return img


def draw_feature_4(image, feature):
    img = draw_rectangle(image, ((feature.y, feature.x), (feature.y + feature.height/2 - 1, feature.x + feature.width/2 - 1)), "red")
    draw_rectangle(img, ((feature.y + feature.height/2, feature.x), (feature.y + feature.height - 1, feature.x + feature.width/2 - 1)), "yellow")
    draw_rectangle(img, ((feature.y, feature.x + feature.width/2), (feature.y + feature.height/2 - 1, feature.x + feature.width - 1)), "yellow")
    img = draw_rectangle(image, ((feature.y + feature.height/2, feature.x + feature.width/2), (feature.y + feature.height - 1, feature.x + feature.width - 1)), "red")
    return img
