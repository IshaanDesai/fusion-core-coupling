"""
Definition of class for implementing boundary conditions
"""
import numpy as np
cimport numpy as np
cimport cython
import math

cdef class Boundary:
    cdef double rho_write
    cdef int nrho
    cdef int ntheta
    cdef double drho
    cdef double dtheta
    cdef double [:, ::1] g_rr
    cdef double [:, ::1] g_rt
    cdef double [:, ::1]  g_tt
    cdef double [::1] rho
    cdef double [::1] theta
    cdef double rho_min
    cdef double rho_max
