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


def init_mms(rmin, rmax, r, theta):
    """
    Initial state based on the mms solution f = sin(2*pi*(r - rmin)/(rmax - rmin))*cost(omega*t)*cos(theta)
    Achieved by inserting t = 0 in the solution.
    :param mesh:
    :param r:
    :param theta:
    :return:
    """
    return math.sin(2*math.pi*(r - rmin)/(rmax - rmin))*math.cos(theta)
