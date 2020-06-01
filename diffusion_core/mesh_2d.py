"""
Module to setup 2D mesh in polar coordinate system
"""
import numpy as np
import math
from config import Config
import enum


class MeshVertexType(enum.Enum):
    """
    Defines vertex types: Physical vertex (CORE), and boundary vertex (GHOST).
    """
    CORE = 0  # physical grid point
    GHOST = 1  # ghost grid point


class Mesh:
    """
    Mesh class used to define a mesh in 2D representing a cross section of a torus geometry of a Tokamak.
    """
    def __init__(self, config_file_name='polar_code_config.json'):
        """

        :param config_file_name:
        """

        self._config = Config(config_file_name)

        self._dims = self._config.get_dims()
        self._rhomin = self._config.get_rhomin()
        self._rhomax = self._config.get_rhomax()
        self._r_spacing = self._config.get_r_spacing()
        self._rho_spacing = self._config.get_rho_spacing()

        self._polar_coords = None  # Initialized in create_mesh function
        self._cart_coords = None  # Initialized in create_mesh function
        self._vertex_type = None  # Initialised in create_mesh function

        self._mesh_indices = None  # Initialized in create_mesh function
        self._mesh_index_of_grid = None  # Initialized in create_mesh function
        self._mesh_index_of_ghost = None  # Initialized in create_mesh function
        self._mesh_i, self._mesh_j = None, None  # Initialized in create_mesh function

        self._mesh_count, self._grid_count, self._ghost_count = 0, 0, 0

        self._create_mesh()

    def _create_mesh(self):
        r_v = (self._rhomax - self._rhomin) / self._r_spacing
        rho_v = 2*math.pi / self._rho_spacing

        print("r_v = {}, rho_v = {}".format(r_v, rho_v))

        # Adding 2 extra radial vertex rows as ghost point layers for boundary conditions
        r_vertices = int(round(r_v)) + 2
        rho_vertices = int(round(rho_v))

        print("r_vertices = {}, rho_vertices = {}".format(r_vertices, rho_vertices))

        print("Polar mesh has {} points".format(r_vertices*rho_vertices))

        self._polar_coords = np.zeros((r_vertices, rho_vertices, self._dims))
        self._cart_coords = np.zeros((r_vertices, rho_vertices, self._dims))
        self._mesh_indices = np.zeros((r_vertices, rho_vertices))
        self._mesh_i = np.zeros(r_vertices*rho_vertices)
        self._mesh_j = np.zeros(r_vertices*rho_vertices)

        # https://stackoverflow.com/questions/6667201/how-to-define-a-two-dimensional-array-in-python
        self._vertex_type = [[0 for x in range(rho_vertices)] for y in range(r_vertices)]

        mesh_index_grid, mesh_index_ghost = [], []

        r_count = self._rhomin - self._r_spacing
        rho_count, k = 0, 0
        for i in range(r_vertices):
            for j in range(rho_vertices):
                self._polar_coords[i, j, k] = r_count
                self._polar_coords[i, j, k+1] = rho_count

                self._cart_coords[i, j, k] = r_count*math.cos(rho_count)
                self._cart_coords[i, j, k+1] = r_count*math.sin(rho_count)

                if r_count < self._rhomin or r_count > self._rhomax:
                    self._vertex_type[i][j] = MeshVertexType.GHOST
                    mesh_index_ghost.append(self._mesh_count)
                    self._ghost_count += 1
                else:
                    self._vertex_type[i][j] = MeshVertexType.CORE
                    mesh_index_grid.append(self._mesh_count)
                    self._grid_count += 1

                self._mesh_indices[i, j] = self._mesh_count
                self._mesh_i[self._mesh_count] = i
                self._mesh_j[self._mesh_count] = j

                rho_count += self._rho_spacing
                self._mesh_count += 1

            r_count += self._r_spacing

        print("Mesh points = {}, Grid Points = {}, Ghost points = {}".format(self._mesh_count, self._grid_count,
                                                                             self._ghost_count))

        self._mesh_index_of_grid = np.array(mesh_index_grid)
        self._mesh_index_of_ghost = np.array(mesh_index_ghost)

    def get_polar_rad(self, ind_i, ind_j):
        return self._polar_coords[ind_i, ind_j, 0]

    def get_polar_rho(self, ind_i, ind_j):
        return self._polar_coords[ind_i, ind_j, 1]

    def get_cart_x(self, ind_i, ind_j):
        return self._cart_coords[ind_i, ind_j, 0]

    def get_cart_y(self, ind_i, ind_j):
        return self._cart_coords[ind_i, ind_j, 1]

    def grid_to_mesh_index(self, index):
        return self._mesh_index_of_grid[index]

    def ghost_to_mesh_index(self, index):
        return self._mesh_index_of_ghost[index]

    def get_n_points_mesh(self):
        return self._mesh_count

    def get_n_points_grid(self):
        return self._grid_count

    def get_n_points_ghost(self):
        return self._ghost_count

    def get_neighbor_index(self, index, x_dir, y_dir):
        neigh_i = self._mesh_i[index] + x_dir
        neigh_j = self._mesh_j[index] + y_dir
        return self._mesh_indices[neigh_i, neigh_j]
