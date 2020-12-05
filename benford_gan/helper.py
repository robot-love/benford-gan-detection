import requests
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
