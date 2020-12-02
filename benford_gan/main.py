from math import log
from scipy.fftpack import dct
from math import ceil
from PIL import Image
import jax.numpy as jnp
import requests
from io import BytesIO


def load_sample_img():
    url = "https://www.byrdie.com/thmb/pr2U7ghfvv3Sz8zJCHWFLT2K55E=/735x0/cdn.cliqueinc.com__cache__posts__274058__" \
          "face-masks-for-pores-274058-1543791152268-main.700x0c-270964ab60624c5ca853057c0c151091-" \
          "d3174bb99f944fc492f874393002bab7.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.load()
    data = jnp.asarray(img, dtype="int32")
    return data[:, :, 0]


class ImageBlockIterator:
    def __init__(self, img: jnp.ndarray):
        w, h = img.shape

        self.wblocks = ceil(w/8)
        self.hblocks = ceil(h/8)

        # self.blocks = jnp.zeros((8, 8, wblocks * hblocks))

        self.img = img

    def __iter__(self):

        self.i = 0
        self.j = 0

        return self

        # return jnp.hstack(a[i:1 + n + i - width:stepsize] for i in range(0, width))

    def __next__(self):

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

        islice = slice(i*8,(i+1)*8,1)
        jslice = slice(j*8,(i+1)*8,1)

        block = self.img[islice,jslice]
        
        self.i = i
        self.j = j
        print(i)
        print(j)
        return block


def load_image(filename, to_grayscale = False):
    img = Image.open(filename)
    if to_grayscale:
        img = img.convert('LA')
    img.thumbnail((256,256))
    img.load()
    data = jnp.asarray(img, dtype = "int32")
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


def main():
    img = load_sample_img()

    sample = ImageBlockIterator(img)

    for block in sample:
        print(dct(block))


if __name__ == '__main__':
    main()