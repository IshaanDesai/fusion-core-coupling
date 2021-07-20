"""
Analytical solution module for the diffusion equation using Bessel functions
"""

import numpy as np
cimport numpy as np
cimport cython
from scipy import special
import math

cdef class Ansol:
    cdef double m
    cdef double ums
    cdef int nrho
    cdef int ntheta
    cdef double drho
    cdef double dtheta
    cdef double [::1] rho
    cdef double [::1] theta
