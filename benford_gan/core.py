from typing import List
import numpy as np
from math import floor, log
from scipy.fftpack import dct
from scipy.optimize import least_squares
from scipy.special import softmax

from config import ZIGZAG_IND_8X8, QTable
from helper import *


class BenfordFeature:
    """
    A data struct for benford features used to detect GAN-generated images
    """

    def __init__(self, feature, frequencies, bases, qtables, label):
        self.feature = feature
        self.frequencies = frequencies
        self.bases = bases
        self.qtables = qtables
        self.label = label

    def __repr__(self):
        rep = f"{'Feature shape: ':>20}{self.feature.shape}\n"\
                f"{'Frequencies: ':>20}{', '.join(map(str, self.frequencies))}\n"\
                f"{'Bases: ':>20}{', '.join(map(str, self.bases))}\n"\
                f"{'Quantization tables: ':>20}{self.qtables}\n"\
                f"{'Label: ':>20}{self.label}"
        
        return rep


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
        x = round(log(abs(ck), b),2)
        fd = floor(abs(ck) / ( b ** floor(x) ))
    return fd


@np.vectorize
def first_digit_alt(num, base):
    n = '%.2E' % abs(num)
    fd = int(n[:1])
    return fd


# @np.vectorize
def dct_pmf(fds, ds):
    # ds = [i for i in range(base-1)]
    pmf = np.histogram(fds, ds)[0]
    num_elements = fds.shape[0]

    # Additive smoothing to prevent zero elements
    for i in range(len(pmf)):
        if pmf[i] == 0:
            pmf[i] += 1
            num_elements += 1

    pmf = pmf / num_elements

    return pmf


def get_image_dct_coefs(img, freqs, qtables, channel = 'lum'):
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
    # options only for luminance and chrominance
    # TODO: select qtable for luminance/chrominance channels
    assert channel in ['lum','chr']

    img = ImageBlockIterator(img)

    inds = [ZIGZAG_IND_8X8[i] for i in freqs]

    dct_coeffs = np.zeros((len(img), len(freqs), len(qtables)))

    for i, qtable in enumerate(qtables):
        table = qtable.ravel()
        for j, block in enumerate(img):
            # transform image block to freq. domain before applying quantization table coefficient
            # Luminance only -> [:,:,0]
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
    ds = [i+0.5 for i in range(base)]
    shape = (base - 1,  fds.shape[1], fds.shape[2])
    pmf = np.zeros(shape)

    for i in range(shape[1]):
        for j in range(shape[2]):
            pmf[:, i, j] = dct_pmf(fds[:, i, j], ds)
            # pmf[:,i,j] = np.histogram(fds[:,i,j], ds)[0] / fds.shape[0]

    return pmf
    

@np.vectorize
def general_benford_pmf(digit, beta, gamma, delta, base):
    """ General, parameterized form of benfords law. """
    try:
        p = beta * log(1 + 1/(gamma + digit**delta), base)
    except ValueError:
        print(f"{beta} - {gamma} - {delta}")
        p = 0
    return p


def benford_pmf(base):
    return np.asarray([log(1+1/(d+1), base) for d in range(base-1)])


def mmse_benford_cost(x, pmf, base, ds, *args, **kwargs) -> float:
    """ Benford pmf fit cost function """
    # ds = [i+1 for i in range(base-1)]
    pfit = general_benford_pmf(ds, x[0], x[1], x[2], base)
    cost = np.linalg.norm((pfit - pmf)**2)
    return cost


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

    djs = div_kullback_leibler(p_hat, p) + div_kullback_leibler(p, p_hat)

    return djs


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
    dkl = sum([p_hat[i]*log(p_hat[i]/p[i]) for i in range(len(p))])
    return dkl


def S(p, q, a):
    assert len(p) == len(q)
    s = sum([q[d]**a / p[d]**(a-1) for d in range(len(p))])
    return s


def div_renyi(p, p_hat, alpha = 0.1):
    """ Renyi divergence. """
    dr = (1/(1/alpha)) * ( log(S(p,p_hat, alpha)) + log(S(p_hat, p, alpha)) )
    return dr


def get_phi(pmf):
    """
    Generate the Benford feature of an image pmf

    Parameters
    ----------
    pfit

    Returns
    -------

    """

    shape = (3, pmf.shape[1], pmf.shape[2])
    phi = np.zeros(shape)

    base = len(pmf[:,0,0])

    for i in range(pmf.shape[1]):
        for j in range(pmf.shape[2]):
            p = pmf[:,i,j]
            ds = [i+1 for i in range(len(p))]
            x0 = [1,1,1]

            kwargs = {
                'pmf': p,
                'base': base,
                'ds': ds
            }

            xs = least_squares(fun = mmse_benford_cost, x0 = x0 , kwargs = kwargs)

            beta = xs.x[0]
            gamma = xs.x[1]
            delta = xs.x[2]

            pfit = general_benford_pmf(ds, beta, gamma, delta, base)

            try:
                phi[:,i,j] = np.array(
                    [
                        div_jensen_shannon(pfit, p), 
                        div_kullback_leibler(pfit, p), 
                        div_renyi(pfit, p)
                    ]
                )
            except ValueError:
                raise

    return phi


def generate_benford_feature(img: np.ndarray, frequencies: List[int], bases: List[int], qtables: List[QTable], label):
    """ Generate a Benford feature for an image based on config """

    # Get DCT Coefficients
    dct_coeffs = get_image_dct_coefs(img, frequencies, [q.matrix for q in qtables])

    features = np.zeros((3, len(frequencies), len(bases), len(qtables)))

    for i,base in enumerate(bases):
        # Get DCT Coefficient First Digits
        fds = dct_coeff_to_first_digit(dct_coeffs, base)

        # Get FD pmf
        pmf = get_dct_fd_pmf(fds, base)

        # Generate benford feature
        phi = get_phi(pmf)
        features[:, :, i, :] = phi

    b = BenfordFeature(features, frequencies, bases, [q.name for q in qtables], label)

    return b