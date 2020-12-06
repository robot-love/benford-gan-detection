"""

"""

from helper import *
from core import *
from config import *

def main():
    img = load_sample_img()


    fds = get_image_fds(img, **args)
    

    first_digit_vec = np.vectorize(first_digit)
    fds_vec = fds.reshape((91 * len(frequencies), 1))
    A = first_digit_vec(fds_vec, b)
    A = A.reshape((91, len(frequencies)))

    hists = [np.histogram(A[:, j],[i for i in range(b)]) / len(fds) for j in range(len(frequencies))]


if __name__ == '__main__':
    main()
