"""
This module implements boundary conditions on the ghost points of a given mesh.
Dirichlet and Neumann boundary conditions can be implemented.
For Neumann boundary conditions, the flux normal is in the inward directions (towards the center of polar grid).
"""
from .mesh_2d import MeshVertexType
import enum


class BoundaryType(enum.Enum):
    """
    Defines vertex types: Physical vertex (CORE), and boundary vertex (GHOST).
    """
    DIRICHLET = 2
    NEUMANN = 3


def set_bnd_vals_dirichlet(mesh, data, field):
    nr, ntheta = mesh.get_n_points_axiswise()
    counter = 0
    for i in range(nr):
        for j in range(ntheta):
            # Point type is a list and not a numpy array so it requires different index accessing
            if mesh.get_point_type(i, j - 1) == MeshVertexType.GHOST:
                field[i, j] = data[counter]
                counter += 1


def set_bnd_vals_neumann(mesh, fluxes, field):
    nr, ntheta = mesh.get_n_points_axiswise()
    dr = mesh.get_r_spacing()
    counter = 0
    for i in range(nr):
        for j in range(ntheta):
            # Point type is a list and not a numpy array so it requires different index accessing
            if mesh.get_point_type(i, j - 1) == MeshVertexType.GHOST:
                # Second order formulation of Neumann BC using two inner points (only valid for Core)
                field[i, j] = (4/3)*field[i-1, j] - (1/3)*field[i-2, j] - (2/3)*dr*fluxes[counter]
                counter += 1
