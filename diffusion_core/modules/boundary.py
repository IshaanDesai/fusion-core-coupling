"""
This module implements boundary conditions on the ghost points of a given mesh.
Dirichlet and Neumann boundary conditions can be implemented.
For Neumann boundary conditions, the flux normal is in the inward directions (towards the center of polar grid).
"""
import enum
import numpy as np


class BoundaryType(enum.Enum):
    """
    Defines vertex types: Physical vertex (CORE), and boundary vertex (GHOST).
    """
    DIRICHLET = 2
    NEUMANN = 3


class Boundary:
    def __init__(self, mesh, data, field, bnd_type, layer_type):
        self._nr, self._ntheta = mesh.get_n_points_axiswise()
        self._dr = mesh.get_r_spacing()

        self._bnd_i, self._bnd_j = [], []
        self._r, self._x, self._y = [], [], []
        self._bnd_type = bnd_type
        self._layer_type = layer_type

        counter = 0
        for i in range(self._nr):
            for j in range(1, self._ntheta + 1):
                mesh_ind = mesh.get_index_from_i_j(i, j - 1)
                r = mesh.get_r(mesh_ind)
                x = mesh.get_x(mesh_ind)
                y = mesh.get_y(mesh_ind)
                point_type = mesh.get_point_type(i, j - 1)
                if point_type == self._layer_type:
                    self._bnd_i.append(i)
                    self._bnd_j.append(j)
                    if self._bnd_type == BoundaryType.DIRICHLET:
                        field[i, j] = data[counter]
                    elif self._bnd_type == BoundaryType.NEUMANN:
                        # Calculate flux from components
                        flux = data[counter, 0]*(x/r) + data[counter, 1]*(y/r)
                        # Modify boundary value by first order evaluation of gradient
                        field[i, j] = field[i + 1, j] + flux*self._dr
                counter += 1

    def set_bnd_vals(self, field, data):
        counter = 0
        if self._bnd_type == BoundaryType.DIRICHLET:
            for i in self._bnd_i:
                for j in self._bnd_j:
                    field[i, j] = data[counter]
                    counter += 1
        elif self._bnd_type == BoundaryType.NEUMANN:
            for i in self._bnd_i:
                for j in self._bnd_j:
                    # Calculate flux from components
                    flux = data[counter, 0]*(self._x[counter]/self._r[counter]) + data[counter, 1]*(self._y[counter]/self._r[counter])
                    # Modify boundary value by first order evaluation of gradient
                    field[i, j] = field[i + 1, j] + flux * self._dr
                    counter += 1

    def get_bnd_vals(self, field):
        bnd_data = []
        for i in self._bnd_i:
            for j in self._bnd_j:
                bnd_data.append(field[i, j])

        return np.array(bnd_data)
