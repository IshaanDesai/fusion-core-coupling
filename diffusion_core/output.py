"""
This module provides functions to output the data in VTK and other formats for viewing and further post processing
For the VTK export the following package is used: https://github.com/paulo-herrera/PyEVTK
"""

from pyevtk.hl import gridToVTK
import numpy as np


def write_vtk(field, t, x, y):
    filename = "field_out_{}".format(t)
    nx, ny = x.shape
    nz = 1
    x_out = np.zeros((nx, ny, nz))
    y_out = np.zeros((nx, ny, nz))
    z_out = np.zeros((nx, ny, nz))
    field_out = np.zeros((nx, ny, nz))

    for i in range(nx):
        for j in range(ny):
            x_out[i, j, 0] = x[i, j]
            y_out[i, j, 0] = y[i, j]
            z_out[i, j, 0] = 0
            field_out[i, j, 0] = field[i, j]

    gridToVTK("./output/"+filename, x_out, y_out, z_out, pointData={"field": field_out})

    print("VTK output written at t = {}".format(t))
    print("Field magnitude = {}".format(field.sum()))
