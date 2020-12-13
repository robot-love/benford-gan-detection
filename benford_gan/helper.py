import requests
from math import log
from PIL import Image
from io import BytesIO
import numpy as np
import string
digs = string.digits + string.ascii_letters


def load_img_from_url(url: str) -> 'numpy array':
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert('YCbCr')
    img.load()
    data = np.array(img)
    return data


def load_image_from_file(filename: str) -> 'numpy array':
    img = Image.open(filename)
    img = img.convert('YCbCr')
    img.load()
    data = np.asarray(img)
    return data


def generate_sample_benford_pmf(base: int, noise_var: float = 0.1) -> np.ndarray:
    """ Generate a noisy Benford distribution """
    x = [i+1 for i in range(base - 1)]
    p = [abs(log(1 + (1/d), base) + np.random.normal(scale = 0.1)) for d in x]
    p = p / np.sum(p)
    return p


def int2base(x, base):
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]
    else:
        sign = 1

    x *= sign
    digits = []

    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)

    if sign < 0:
        digits.append('-')

    digits.reverse()

    return ''.join(digits)