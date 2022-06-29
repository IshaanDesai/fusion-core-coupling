"""
Definition of class for implementing boundary conditions
"""
import numpy as np
cimport numpy as np
cimport cython
import math

cdef class Boundary:
    cdef double rho_min, rho_max, rho_write, drho, dtheta
    cdef int nrho, ntheta
    cdef double [:, ::1] g_rr, g_rt, g_tt
    cdef double [::1] rho, theta
