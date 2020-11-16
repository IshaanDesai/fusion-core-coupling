"""
This module provides functions to output the data in VTK and other formats for viewing and further post processing
For the VTK export the following package is used: https://github.com/paulo-herrera/PyEVTK
"""

from pyevtk.hl import gridToVTK
import numpy as np
import csv


def write_vtk(field, xpol, ypol, t):
    filename = "field_out_{}".format(t)
    nrho, ntheta = 100, 400
    nz = 1
    x_out = np.zeros((ntheta, nrho, nz))
    y_out = np.zeros((ntheta, nrho, nz))
    z_out = np.zeros((ntheta, nrho, nz))
    point_type = np.zeros((ntheta, nrho, nz))
    field_out = np.zeros((ntheta, nrho, nz))

    for i in range(ntheta):
        for j in range(nrho):
            x_out[i, j, 0] = xpol[i, j]
            y_out[i, j, 0] = ypol[i, j]
            z_out[i, j, 0] = 0
            field_out[i, j, 0] = field[i, j]

    gridToVTK("./output/"+filename, x_out, y_out, z_out, pointData={"field": field_out, "type": point_type})


def write_csv(field, xpol, ypol, n):
    nrho, ntheta = 100, 400
    with open('./output/polar_'+str(n)+'.csv', mode='w') as file:
        file_writer = csv.writer(file, delimiter=',')
        for i in range(ntheta):
            for j in range(nrho):
                file_writer.writerow([xpol[i, j], ypol[i, j], field[i, j]])


def write_custom_csv(coords):
    with open('./output/custom.csv', mode='w') as file:
        file_writer = csv.writer(file, delimiter=',')
        for coord in coords:
            file_writer.writerow([coord[0], coord[1]])
