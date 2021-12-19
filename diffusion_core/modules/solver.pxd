"""
Definition of class for implementing solver schemes
"""
import numpy as np
cimport numpy as np
cimport cython
import math

cdef class Diffusion2D:
        cdef double drho
        cdef double dtheta
        cdef int nrho
        cdef int ntheta

        cdef double [::1] rho
        cdef double [::1] theta

        cdef double [:, ::1] jac
        cdef double [:, ::1] g_rr
        cdef double [:, ::1] g_rt
        cdef double [:, ::1] g_tt
