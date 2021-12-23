"""
Definition of class for implementing solver schemes
"""
import numpy as np
cimport numpy as np
cimport cython
import math

cdef class Diffusion2D:
        cdef double drho, dtheta
        cdef int nrho, ntheta
        cdef double [::1] rho, theta
        cdef double [:, ::1] jac, g_rr, g_rt, g_tt
