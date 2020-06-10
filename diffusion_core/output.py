"""
This module provides functions to output the data in VTK and other formats for viewing and further post processing
For the VTK export the following package is used: https://github.com/paulo-herrera/PyEVTK
"""

from pyevtk.hl import gridToVTK
import numpy as np
from mesh_2d import Mesh, MeshVertexType


def write_vtk(field, t):
    filename = "field_out_{}".format(t)
    problem_config_file = "diffusion-coupling-config.json"
    mesh = Mesh(problem_config_file)

    nx, ny = mesh.get_n_points_axiswise()
    nz = 1
    x_out = np.zeros((nx, ny, nz))
    y_out = np.zeros((nx, ny, nz))
    z_out = np.zeros((nx, ny, nz))
    point_type = np.zeros((nx, ny, nz))
    field_out = np.zeros((nx, ny, nz))

    counter = 0
    for i in range(nx):
        for j in range(ny):
            x_out[i, j, 0] = mesh.get_x(counter)
            y_out[i, j, 0] = mesh.get_y(counter)
            z_out[i, j, 0] = 0
            if mesh.get_point_type(i, j) == MeshVertexType.GHOST:
                point_type[i, j, 0] = 1
            elif mesh.get_point_type(i, j) == MeshVertexType.CORE:
                point_type[i, j, 0] = 0
            field_out[i, j, 0] = field[i, j]
            counter += 1

    gridToVTK("./output/"+filename, x_out, y_out, z_out, pointData={"field": field_out, "type": point_type})

    print("VTK output written at t = {}".format(t))
    print("Field magnitude = {}".format(field.sum()))
