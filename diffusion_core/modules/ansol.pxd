"""
Analytical solution module for the diffusion equation using Bessel functions
"""

import numpy as np
cimport numpy as np
cimport cython
from scipy import special
import math

cdef class Ansol:
    cdef double m, ums, drho, dtheta
    cdef int nrho, ntheta
    cdef double [::1] rho, theta
