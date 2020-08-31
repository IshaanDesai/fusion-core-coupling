"""
This module provides two initialization states for the polar grid:
1. Gaussian blob initialization
2. Custom initialization based on a chosen Ansatz solution for "Method of Manufactured Solutions"
"""
import math


def gaussian_blob(pos, wblob, coord):
    """
    Function to define a Gaussian blob as initial state of a Diffusion problem
    :param pos:
    :param wblob:
    :param coord:
    :return:
    """
    exponent = -1.0 * (coord - pos) * (coord - pos) / (wblob * wblob)
    gaussian = math.exp(exponent)
    return gaussian


