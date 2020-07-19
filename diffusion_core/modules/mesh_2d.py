"""
Module to setup 2D mesh in polar coordinate system
"""
import numpy as np
import math
from .config import Config
import enum
import logging


class MeshVertexType(enum.Enum):
    """
    Defines vertex types: Physical vertex (CORE), and boundary vertex (GHOST).
    """
    CORE = 0  # physical grid point
    GHOST_WALL = 1  # ghost grid point on outer edge
    GHOST_CORE = 2  # ghost grid point on inner edge


class Mesh:
    """
    Mesh class used to define a mesh in 2D representing a cross section of a torus geometry of a Tokamak.
    """
    def __init__(self, config):
        """
        :param config:
        """
        self.logger = logging.getLogger('main.mesh_2d.Mesh')

        self._dims = config.get_dims()
        self._rmin = config.get_rmin()
        self._rmax = config.get_rmax()
        self._r_points = config.get_r_points()
        self._theta_points = config.get_theta_points()

        self._polar_coords_r = None  # Initialized in create_mesh function
        self._polar_coords_theta = None  # Initialized in create_mesh function
        self._cart_coords_x = None  # Initialized in create_mesh function
        self._cart_coords_y = None  # Initialized in create_mesh function
        self._vertex_type = None  # Initialised in create_mesh function

        self._mesh_index = None  # Initialized in create_mesh function
        self._mesh_index_of_grid = None  # Initialized in create_mesh function
        self._mesh_index_of_ghost = None  # Initialized in create_mesh function
        self._mesh_i, self._mesh_j = None, None  # Initialized in create_mesh function

        self._mesh_count, self._grid_count, self._ghostcore_count, self._ghostwall_count = 0, 0, 0, 0

        self._create_mesh()

    def _create_mesh(self):
        self._r_spacing = (self._rmax - self._rmin) / (self._r_points - 1)
        self._theta_spacing = 2*math.pi / self._theta_points

        self.logger.info('r_spacing = %d, theta_spacing = %d', self._r_spacing, self._theta_spacing)
        self.logger.info('Polar mesh has %d points', self._r_points*self._theta_points)

        self._polar_coords_r = np.zeros((self._r_points, self._theta_points))
        self._polar_coords_theta = np.zeros((self._r_points, self._theta_points))
        self._cart_coords_x = np.zeros((self._r_points, self._theta_points))
        self._cart_coords_y = np.zeros((self._r_points, self._theta_points))
        self._mesh_index = np.zeros((self._r_points, self._theta_points), dtype=int)
        self._mesh_i = np.zeros(self._r_points*self._theta_points, dtype=int)
        self._mesh_j = np.zeros(self._r_points*self._theta_points, dtype=int)

        # https://stackoverflow.com/questions/6667201/how-to-define-a-two-dimensional-array-in-python
        self._vertex_type = [[0 for x in range(self._theta_points)] for y in range(self._r_points)]

        mesh_index_grid, mesh_index_ghostcore, mesh_index_ghostwall = [], [], []

        r_val = self._rmin
        for i in range(self._r_points):
            theta_val = 0
            for j in range(self._theta_points):
                self._polar_coords_r[i, j] = r_val
                self._polar_coords_theta[i, j] = theta_val

                self._cart_coords_x[i, j] = r_val*math.cos(theta_val)
                self._cart_coords_y[i, j] = r_val*math.sin(theta_val)

                if r_val <= self._rmin:
                    self._vertex_type[i][j] = MeshVertexType.GHOST_CORE
                    mesh_index_ghostcore.append(self._mesh_count)
                    self._ghostcore_count += 1
                elif r_val >= self._rmax:
                    self._vertex_type[i][j] = MeshVertexType.GHOST_WALL
                    mesh_index_ghostwall.append(self._mesh_count)
                    self._ghostwall_count += 1
                else:
                    self._vertex_type[i][j] = MeshVertexType.CORE
                    mesh_index_grid.append(self._mesh_count)
                    self._grid_count += 1

                self._mesh_index[i, j] = self._mesh_count
                self._mesh_i[self._mesh_count] = i
                self._mesh_j[self._mesh_count] = j

                theta_val += self._theta_spacing
                self._mesh_count += 1

            r_val += self._r_spacing

        self._ghost_count = self._ghostcore_count + self._ghostwall_count

        self.logger.info('Total mesh points = %d, Grid Points = %d, Ghost points = %d', self._mesh_count,
                         self._grid_count, self._ghost_count)

        self._mesh_index_of_grid = np.array(mesh_index_grid)
        self._mesh_index_of_ghostcore = np.array(mesh_index_ghostcore)
        self._mesh_index_of_ghostwall = np.array(mesh_index_ghostwall)

    def get_i_j_from_index(self, index):
        return self._mesh_i[index], self._mesh_j[index]

    def get_index_from_i_j(self, ind_i, ind_j):
        return self._mesh_index[ind_i, ind_j]

    def get_x(self, index):
        i, j = self.get_i_j_from_index(index)
        return self._cart_coords_x[i, j]

    def get_y(self, index):
        i, j = self.get_i_j_from_index(index)
        return self._cart_coords_y[i, j]

    def get_r(self, index):
        i, j = self.get_i_j_from_index(index)
        return self._polar_coords_r[i, j]

    def get_theta(self, index):
        i, j = self.get_i_j_from_index(index)
        return self._polar_coords_theta[i, j]

    def grid_to_mesh_index(self, index):
        return self._mesh_index_of_grid[index]

    def ghostcore_to_mesh_index(self, index):
        return self._mesh_index_of_ghostcore[index]

    def ghostwall_to_mesh_index(self, index):
        return self._mesh_index_of_ghostwall[index]

    def get_n_points_mesh(self):
        return self._mesh_count

    def get_n_points_grid(self):
        return self._grid_count

    def get_n_points_ghost(self):
        return self._ghost_count

    def get_n_points_ghostcore(self):
        return self._ghostcore_count

    def get_n_points_ghostwall(self):
        return self._ghostwall_count

    def get_n_points_axiswise(self):
        return self._r_points, self._theta_points

    def get_point_type(self, ind_i, ind_j):
        return self._vertex_type[ind_i][ind_j]

    def get_r_spacing(self):
        return self._r_spacing

    def get_theta_spacing(self):
        return self._theta_spacing
