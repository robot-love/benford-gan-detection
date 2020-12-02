from math import log
from scipy.fftpack import dct
# from scipy import misc
from PIL import Image
import numpy as np


SAMPLEIMG = "C:\\Users\\sghavam\\Pictures\\jamill-del-rosario-HrOwk2kX9g0-unsplash.jpg"


class ImageBlockIterator:
    def __init__(self, img):
        # Downsample

        self.img = img

    def __iter__(self):
        pass


def load_image(filename, to_grayscale = False):
    img = Image.open(filename)
    if to_grayscale:
        img = img.convert('LA')
    img.thumbnail((256,256))
    img.load()
    data = np.asarray(img, dtype = "int32")
    return data[:, :, 0]


def first_digit(ck, b):
    """
    Get first digit of DCT coefficient from the kth block

    Parameters
    ----------
    ck
    b

    Returns
    -------

    """
    fd = abs(ck) / b ^ (log(ck, b))
    return fd


