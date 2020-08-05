"""
This module implements boundary conditions on the boundary points of a given mesh.
Dirichlet and Neumann boundary conditions can be implemented.
For Neumann boundary conditions, the flux normal is in the inward directions (towards the center of polar grid).

NOTE: Neumann boundary conditions are implemented with the assumption that only Wall Boundary points will have
      Neumann conditions.
"""
import enum
import numpy as np


class BoundaryType(enum.Enum):
    """
    Defines boundary types: DIRICHLET = Dirichlet, NEUMANN_FO, NEUMANN_SO = Neumann first and second order.
    """
    DIRICHLET = 2
    NEUMANN_FO = 3
    NEUMANN_SO = 4


class Boundary:
    def __init__(self, mesh, data, field, bnd_type, layer_type):
        self._nr, self._ntheta = mesh.get_n_points_axiswise()
        self._dr = mesh.get_r_spacing()

        self._bnd_inds = []
        self._r, self._x, self._y = [], [], []
        self._bnd_type = bnd_type
        self._layer_type = layer_type

        counter = 0
        for i in range(self._nr):
            for j in range(self._ntheta):
                mesh_ind = mesh.get_index_from_i_j(i, j)
                r = mesh.get_r(mesh_ind)
                x = mesh.get_x(mesh_ind)
                y = mesh.get_y(mesh_ind)
                point_type = mesh.get_point_type(i, j)
                if point_type == self._layer_type:
                    self._bnd_inds.append([i, j])
                    if self._bnd_type == BoundaryType.DIRICHLET:
                        field[i, j] = data[counter]
                        counter += 1
                    elif self._bnd_type == BoundaryType.NEUMANN_FO:
                        # Calculate flux from components
                        flux = data[counter, 0] * (self._x[counter] / self._r[counter]) + data[counter, 1] * (
                            self._y[counter] / self._r[counter])
                        # Modify boundary value by first order evaluation of gradient
                        field[i, j] = field[i - 1, j] + flux*self._dr
                        counter += 1
                    elif self._bnd_type == BoundaryType.NEUMANN_SO:
                        # Calculate flux from components
                        flux = data[counter, 0] * (self._x[counter] / self._r[counter]) + data[counter, 1] * (
                            self._y[counter] / self._r[counter])
                        # Modify boundary value by second order evaluation of gradient
                        field[i, j] = (4/3)*field[i - 1, j] - (1/3)*field[i - 2, j] - (2/3)*self._dr*flux
                        counter += 1

                self._r.append(r)
                self._x.append(x)
                self._y.append(y)

    def set_bnd_vals(self, field, data):
        counter = 0
        if self._bnd_type == BoundaryType.DIRICHLET:
            for inds in self._bnd_inds:
                field[inds[0], inds[1]] = data[counter]
                counter += 1
        elif self._bnd_type == BoundaryType.NEUMANN_FO:
            counter = 0
            for inds in self._bnd_inds:
                # Calculate flux from components
                flux = data[counter, 0]*(self._x[counter]/self._r[counter]) + data[counter, 1]*(self._y[counter]/self._r[counter])
                # Modify boundary value by first order evaluation of gradient
                field[inds[0], inds[1]] = field[inds[0] - 1, inds[1]] + flux*self._dr
                counter += 1
        elif self._bnd_type == BoundaryType.NEUMANN_SO:
            counter = 0
            for inds in self._bnd_inds:
                # Calculate flux from components
                flux = data[counter, 0]*(self._x[counter]/self._r[counter]) + data[counter, 1]*(self._y[counter]/self._r[counter])
                # Modify boundary value by second order evaluation of gradient
                field[inds[0], inds[1]] = (4/3)*field[inds[0] - 1, inds[1]] - (1/3)*field[inds[0] - 2, inds[1]] - (2/3)*self._dr*flux
                counter += 1

    def get_bnd_vals(self, field):
        bnd_data = []
        # Gets Dirichlet values and returns them for coupling
        for inds in self._bnd_inds:
            bnd_data.append(field[inds[0], inds[1]])

        return np.array(bnd_data)
