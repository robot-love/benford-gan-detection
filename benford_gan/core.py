from typing import List
import numpy as np
from math import floor, log
from scipy.fftpack import dct

from benford_gan.config import ZIGZAG_IND_8X8


class ImageBlockIterator:
    """
    A class to iterate over an image in blocks of NxN. In order to prevent undesirable statistical traces from incorrect
    image padding, axes are cropped to the nearest multiple of the block size in that dimension.
    """
    def __init__(self, img: np.ndarray, block_size = 8):
        # blocks are square.
        self.block_size = block_size
        self.wblocks = floor(img.shape[0] / self.block_size)
        self.hblocks = floor(img.shape[1] / self.block_size)

        self.width = self.wblocks*self.block_size
        self.height = self.hblocks*self.block_size

        self.img = img[0:self.width, 0:self.height, :]

    def __len__(self):
        return self.wblocks * self.hblocks

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

        if i > self.wblocks:
            i = 0
            j = self.j + 1
        else:
            j = self.j

        islice = slice(i * self.block_size, (i + 1) * self.block_size, 1)
        jslice = slice(j * self.block_size, (j + 1) * self.block_size, 1)

        block = self.img[islice, jslice]

        if len(block) == 0 or i > self.wblocks and self.j >= self.hblocks:
            self.i = 0
            self.j = 0
            raise StopIteration

        self.i = i
        self.j = j

        return block


@np.vectorize
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
    if ck == 0:
        fd = 0
    else:
        fd = floor(abs(ck) / (b ** floor((log(abs(ck), b)))))
    return fd


def get_image_fds(img, freqs, qtables):
    """
    Generate a NxMxK

    Parameters
    ----------
    img
    freqs
    qtables

    Returns
    -------

    """
    assert all(isinstance(x, int) for x in freqs)
    assert max(freqs) < 64
    assert min(freqs) > -1

    img = ImageBlockIterator(img)

    inds = [ZIGZAG_IND_8X8[i] for i in freqs]

    fds = np.zeros((len(freqs), len(img), len(qtables)))

    for i, qtable in enumerate(qtables):
        table = qtable.ravel()
        for j, block in enumerate(img):
            # transform image block to freq. domain before applying quantization table coefficient
            tform = dct(block).ravel()
            fds[:, j, i] = np.asarray([round(tform[k]/table[k]) for k in inds])

    return np.asarray(fds)


def get_dct_fd_pmf(fds: 'numpy array', base) -> List[np.ndarray]:
    """

    Parameters
    ----------
    fds

    Returns
    -------

    """
    # vectorize first digits
    fds_vec = fds.reshape((fds.shape[0] * len(freqs), 1))

    A = first_digit(fds_vec, base)
    # A = A.reshape((fds.shape[0], len(freqs)))

    hists = [np.histogram(A[:, j], [i for i in range(b)][1:]) for j in range(len(frequencies))]
    hists = [[h[0] / len(fds), h[1]] for h in hists]


def general_benford_pmf(base):
    pass


def fit_pmf_to_benford(p: 'numpy array') -> 'numpy array':
    pass


def div_jensen_shannon(p, p_hat):
    """
    Jensen-Shannon divergence measure of 2 probability mass functions. This is a symmetrized version of Kullback-Leibler
    divergence.

    Parameters
    ----------
    p
    p_hat

    Returns
    -------

    """

    pass


def div_kullback_leibler(p, p_hat):
    """
    Kullback-Leibler divergence of 2 prob. mass functions.

    Parameters
    ----------
    p
    p_hat

    Returns
    -------

    """
    assert len(p) == len(p_hat)

    dkl = 0.0

    for i in range(len(p)):
        dkl += p_hat[i]*log(p_hat[i]/p[i])

    return dkl


def div_renyi(p, p_hat):
    pass


def generate_benford_feature(pfit):
    """
    Generate the Benford feature of an image pmf

    Parameters
    ----------
    pfit

    Returns
    -------

    """
    phi = None

    return phi

