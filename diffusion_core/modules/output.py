"""
This module provides functions to output the data in VTK and other formats for viewing and further post processing
For the VTK export the following package is used: https://github.com/paulo-herrera/PyEVTK
"""

from pyevtk.hl import gridToVTK
import numpy as np
from .mesh_2d import MeshVertexType


def write_vtk(field, mesh, t):
    filename = "field_out_{}".format(t)
    nr, ntheta = mesh.get_n_points_axiswise()
    nz = 1
    x_out = np.zeros((nr, ntheta, nz))
    y_out = np.zeros((nr, ntheta, nz))
    z_out = np.zeros((nr, ntheta, nz))
    point_type = np.zeros((nr, ntheta, nz))
    field_out = np.zeros((nr, ntheta, nz))

    counter = 0
    for i in range(nr):
        for j in range(ntheta):
            x_out[i, j, 0] = mesh.get_x(counter)
            y_out[i, j, 0] = mesh.get_y(counter)
            z_out[i, j, 0] = 0
            if mesh.get_point_type(i, j) == MeshVertexType.GHOST:
                point_type[i, j, 0] = 1
            elif mesh.get_point_type(i, j) == MeshVertexType.CORE:
                point_type[i, j, 0] = 0
            field_out[i, j, 0] = field[i, j+1]
            counter += 1

    gridToVTK("./output/"+filename, x_out, y_out, z_out, pointData={"field": field_out, "type": point_type})


