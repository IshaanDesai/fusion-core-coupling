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
        self._rho_points = self._config.get_rho_points()

        self._polar_coords = None  # Initialized in create_mesh function
        self._cart_coords = None  # Initialized in create_mesh function
        self._vertex_type = None  # Initialised in create_mesh function

        self._mesh_index = None  # Initialized in create_mesh function
        self._mesh_index_of_grid = None  # Initialized in create_mesh function
        self._mesh_index_of_ghost = None  # Initialized in create_mesh function
        self._mesh_i, self._mesh_j = None, None  # Initialized in create_mesh function

        self._mesh_count, self._grid_count, self._ghost_count = 0, 0, 0
        self._nx, self._ny = 0, self._config.get_rho_points()

        self._create_mesh()

    def _create_mesh(self):
        r_v = (self._rhomax - self._rhomin) / self._r_spacing
        rho_spacing = 2*math.pi / self._rho_points

        print("r_v = {}, rho_v = {}".format(r_v, rho_spacing))

        # Adding 2 extra radial vertex rows as ghost point layers for boundary conditions
        r_vertices = int(round(r_v)) + 2

        self._nx = r_vertices

        print("r_vertices = {}, rho_vertices = {}".format(r_vertices, self._rho_points))

        print("Polar mesh has {} points".format(r_vertices*self._rho_points))

        self._polar_coords_r = np.zeros((r_vertices, self._rho_points))
        self._polar_coords_rho = np.zeros((r_vertices, self._rho_points))
        self._cart_coords_x = np.zeros((r_vertices, self._rho_points))
        self._cart_coords_y = np.zeros((r_vertices, self._rho_points))
        self._mesh_index = np.zeros((r_vertices, self._rho_points))
        self._mesh_i = np.zeros(r_vertices*self._rho_points)
        self._mesh_j = np.zeros(r_vertices*self._rho_points)

        # https://stackoverflow.com/questions/6667201/how-to-define-a-two-dimensional-array-in-python
        self._vertex_type = [[0 for x in range(self._rho_points)] for y in range(r_vertices)]

        mesh_index_grid, mesh_index_ghost = [], []

        r_count = self._rhomin - self._r_spacing
        rho_count, k = 0, 0
        for i in range(r_vertices):
            for j in range(self._rho_points):
                self._polar_coords_r[i, j] = r_count
                self._polar_coords_rho[i, j] = rho_count

                self._cart_coords_x[i, j] = r_count*math.cos(rho_count)
                self._cart_coords_y[i, j] = r_count*math.sin(rho_count)

                if r_count < self._rhomin or r_count > self._rhomax:
                    self._vertex_type[i][j] = MeshVertexType.GHOST
                    mesh_index_ghost.append(self._mesh_count)
                    self._ghost_count += 1
                else:
                    self._vertex_type[i][j] = MeshVertexType.CORE
                    mesh_index_grid.append(self._mesh_count)
                    self._grid_count += 1

                self._mesh_index[i, j] = self._mesh_count
                self._mesh_i[self._mesh_count] = i
                self._mesh_j[self._mesh_count] = j

                rho_count += rho_spacing
                self._mesh_count += 1

            r_count += self._r_spacing

        print("Mesh points = {}, Grid Points = {}, Ghost points = {}".format(self._mesh_count, self._grid_count,
                                                                             self._ghost_count))

        self._mesh_index_of_grid = np.array(mesh_index_grid)
        self._mesh_index_of_ghost = np.array(mesh_index_ghost)

    def get_cart_coords(self):
        return self._cart_coords_x, self._cart_coords_y

    def get_polar_coords(self):
        return self._polar_coords_r, self._polar_coords_rho

    def get_i_j_from_index(self, index):
        return int(self._mesh_i[index]), int(self._mesh_j[index])

    def get_x(self, index):
        i, j = self.get_i_j_from_index(index)
        return self._cart_coords_x[i, j]

    def get_y(self, index):
        i, j = self.get_i_j_from_index(index)
        return self._cart_coords_y[i, j]

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

    def get_n_points_axiswise(self):
        return self._nx, self._ny

    def get_point_type(self, ind_i, ind_j):
        return self._vertex_type[ind_i][ind_j]

    def get_r_spacing(self):
        return self._r_spacing

    def get_rho_spacing(self):
        return 2*math.pi / self._rho_points

    def get_neighbor_index(self, index, x_dir, y_dir):
        neigh_i = int(self._mesh_i[index] + x_dir)
        neigh_j = int(self._mesh_j[index] + y_dir)

        # Handling periodicity in the angular direction
        if neigh_j == self._ny:
            return neigh_i, 0
        else:
            return neigh_i, neigh_j

    def polar_to_cartesian(self, r, rho):
        x_coord = r*math.cos(rho)
        y_coord = r*math.sin(rho)
        if x_coord < self._r_spacing:
            x_coord = 0
        if y_coord < self.get_rho_spacing():
            y_coord = 0

        return x_coord, y_coord
