import requests
from math import log
from PIL import Image
from io import BytesIO
import numpy as np


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
    p = [log(1 + (1/d), base) + np.random.normal(scale = 0.1) for d in x]