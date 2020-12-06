from typing import List
import numpy as np
from math import floor, log
from scipy.fftpack import dct

from config import ZIGZAG_IND_8X8


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


        if i == self.wblocks:
            i = 0
            j = self.j + 1
        else:
            j = self.j

        if self.j == self.hblocks - 1:
            self.i = 0
            self.j = 0
            raise StopIteration

        islice = slice(i * self.block_size, (i + 1) * self.block_size, 1)
        jslice = slice(j * self.block_size, (j + 1) * self.block_size, 1)

        block = self.img[islice, jslice]

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


def get_image_dct_coefs(img, freqs, qtables):
    """
    Generate a 3D matrix of first digits of quantized DCT coefficients.

    Parameters
    ----------
    img: np.ndarray
        NxMx3 RGB color image to be analyzed for natural-ness.
    freqs: iterable
        Nx1 DCT frequency indices to include in the feature vector
    qtables: np.ndarray
        8x8 JPEG quantization table

    Returns
    -------
    np.ndarray
        LxMxN 3D array of first digits. 
        
        Axes: 
        - i: image block
        - j: frequency
        - k: quantization tables

    """
    assert all(isinstance(x, int) for x in freqs)
    assert max(freqs) < 64
    assert min(freqs) > -1

    img = ImageBlockIterator(img)

    inds = [ZIGZAG_IND_8X8[i] for i in freqs]

    dct_coeffs = np.zeros((len(img), len(freqs), len(qtables)))

    for i, qtable in enumerate(qtables):
        table = qtable.ravel()
        for j, block in enumerate(img):
            # transform image block to freq. domain before applying quantization table coefficient
            # Luminance only! -> [:,:,0]
            tform = dct(block[:,:,0]).ravel()

            dct_coeffs[j, :, i] = np.asarray([round(tform[k]/table[k]) for k in inds])

    return dct_coeffs


def dct_coeff_to_first_digit(dct_coeffs, base):
    """Get first digits from DCT coefficients

    Args:
        dct_coeffs ([type]): [description]
        base ([type]): [description]
    """
    fds = np.zeros_like(dct_coeffs)

    for i in range(dct_coeffs.shape[1]):
        for j in range(dct_coeffs.shape[2]):
            fds[:,i,j] = first_digit(dct_coeffs[:,i,j], base)

    return fds


def get_dct_fd_pmf(fds: np.ndarray, base: int) -> np.ndarray:
    """

    Parameters
    ----------
    fds: np.ndarray
        3D array of first digits.
        Axes:
        - 0: Image blocks
        - 1: DCT frequencies
        - 2: Quantization tables
    base: int
        Base of first digits

    Returns
    -------
    np.ndarray
        Probability mass function of all digits in range [0, base-1] for all frequencies and Q tables.

    """
    shape = (base - 1,  fds.shape[1], fds.shape[2]) 

    pmf = np.zeros(shape)

    for i in range(shape[1]):
        for j in range(shape[2]):
            pmf[:,i,j] = np.histogram(fds[:,i,j], [i for i in range(base)])[0] / fds.shape[0]

    return pmf
    

@np.vectorize
def general_benford_pmf(digit, beta, gamma, delta, base):
    """ General, parameterized form of benfords law. """
    p = beta * log(1 + 1/(gamma + digit**delta), base=base)
    return p


def fit_pmf_to_benford(p: np.ndarray) -> np.ndarray:
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


def div_kullback_leibler(p: np.ndarray, p_hat:np.ndarray):
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

