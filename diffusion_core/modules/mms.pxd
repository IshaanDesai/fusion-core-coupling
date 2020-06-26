"""
Definition of class for implementing Method of Manufactured Solutions
"""
import numpy as np
cimport numpy as np
cimport cython
import math

cdef class MMS:
    cdef double rmin
    cdef double rmax
    cdef double diffc
    cdef int nr
    cdef int ntheta
    cdef double dr
    cdef double dtheta