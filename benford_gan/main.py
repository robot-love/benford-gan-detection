from math import log
from scipy.fftpack import dct
from math import ceil, floor
from PIL import Image
import numpy as np
import requests
from io import BytesIO


ZIGZAG_IND_8X8 = [
    0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,49,56,57,50,43,
    36,29,22,15,23,30,37,44,51,58,59,60,53,46,39,47,54,61,62,55,63
]



def load_sample_img():
    url = "https://www.byrdie.com/thmb/pr2U7ghfvv3Sz8zJCHWFLT2K55E=/735x0/cdn.cliqueinc.com__cache__posts__274058__" \
          "face-masks-for-pores-274058-1543791152268-main.700x0c-270964ab60624c5ca853057c0c151091-" \
          "d3174bb99f944fc492f874393002bab7.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.load()
    data = np.asarray(img, dtype = "int32")
    return data[:, :, 0]


class ImageBlockIterator:
    def __init__(self, img: np.ndarray, block_size = 8):
        self.block_size = block_size
        self.wblocks = ceil(img.shape[0] / self.block_size)
        self.hblocks = ceil(img.shape[1] / self.block_size)
        self.img = img

    def __iter__(self):
        self.i = 0
        self.j = 0
        return self

    def __next__(self):
        """
        The real reason for this class.

        Returns
        -------

        """
        i = self.i + 1

        if i > self.wblocks and self.j >= self.hblocks:
            self.i = 0
            self.j = 0
            raise StopIteration

        if i > self.wblocks:
            i = 0
            j = self.j + 1
        else:
            j = self.j

        islice = slice(i * self.block_size, (i + 1) * self.block_size, 1)
        jslice = slice(j * self.block_size, (j + 1) * self.block_size, 1)

        block = self.img[islice, jslice]

        if len(block) == 0:
            raise StopIteration

        self.i = i
        self.j = j

        return block


def load_image(filename, to_grayscale = False):
    img = Image.open(filename)
    if to_grayscale:
        img = img.convert('LA')
    img.thumbnail((256, 256))
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


def get_image_fds(img, freqs, *args, **kwargs):

    assert all(isinstance(x, int) for x in freqs)
    assert max(freqs) < 65
    assert min(freqs) > -1

    img = ImageBlockIterator(img)

    inds = [ZIGZAG_IND_8X8[i] for i in freqs]

    fds = []

    for i,block in enumerate(img):
        tform = dct(block).ravel()
        fds.append([tform[j] for j in inds])

    return fds


def main():
    img = load_sample_img()

    sample = ImageBlockIterator(img)

    #
    args = {
        'freqs': [0, 1, 2, 3, 5, 10],
        'bases': [10, 20, 40],
        'steps': []
    }

    fds = get_image_fds(img, **args)

    for block_digits in fds:
        for freq in block_digits:



if __name__ == '__main__':
    main()
