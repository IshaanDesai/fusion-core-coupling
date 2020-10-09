"""
This module provides functions to output the data in VTK and other formats for viewing and further post processing
For the VTK export the following package is used: https://github.com/paulo-herrera/PyEVTK
"""

from pyevtk.hl import gridToVTK
import numpy as np
import csv


def write_vtk(field, xpol, ypol, t):
    filename = "field_out_{}".format(t)
    nrho, ntheta = xpol.size, ypol.size
    nz = 1
    x_out = np.zeros((nrho, ntheta, nz))
    y_out = np.zeros((nrho, ntheta, nz))
    z_out = np.zeros((nrho, ntheta, nz))
    point_type = np.zeros((nrho, ntheta, nz))
    field_out = np.zeros((nrho, ntheta, nz))

    counter = 0
    for i in range(nrho):
        for j in range(ntheta):
            x_out[i, j, 0] = 
            y_out[i, j, 0] = mesh.get_y(counter)
            z_out[i, j, 0] = 0
            if mesh.get_point_type(i, j) == MeshVertexType.BC_CORE or mesh.get_point_type(i, j) == MeshVertexType.BC_WALL:
                point_type[i, j, 0] = 1
            elif mesh.get_point_type(i, j) == MeshVertexType.GRID:
                point_type[i, j, 0] = 0
            field_out[i, j, 0] = field[i, j]
            counter += 1

    gridToVTK("./output/"+filename, x_out, y_out, z_out, pointData={"field": field_out, "type": point_type})


def write_csv(field, mesh, n):
    nr, ntheta = mesh.get_n_points_axiswise()
    counter = 0
    with open('./output/polar_'+str(n)+'.csv', mode='w') as file:
        file_writer = csv.writer(file, delimiter=',')
        for i in range(nr):
            for j in range(ntheta):
                file_writer.writerow([mesh.get_x(counter), mesh.get_y(counter), field[i, j]])
                counter += 1

