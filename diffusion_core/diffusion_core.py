"""
Code to simulate diffusion in a polar coordinate system replicating a gyrokinetics fusion code (reactor core physics)
"""
import numpy as np
from mesh_2d import Mesh, MeshVertexType
from output import write_vtk
from config import Config
from boundary import set_bnd_vals
import math
import time


def gaussian_blob(pos, wblob, coord):
    """
    Function to define a Gaussian blob as initial state of a Diffusion problem
    :param pos:
    :param wblob:
    :param coord:
    :return:
    """
    exponent = -1.0 * (coord - pos) * (coord - pos) / (wblob * wblob)
    gaussian = math.exp(exponent)
    return gaussian


class Diffusion:
    def __init__(self):
        self._file = None

    def solve_diffusion(self):
        # Read initial conditions from a JSON config file
        problem_config_file = "diffusion-coupling-config.json"
        config = Config(problem_config_file)

        # Mesh setup
        mesh = Mesh(problem_config_file)
        nr, ntheta = mesh.get_n_points_axiswise()
        rmin, rmax = config.get_rmin(), config.get_rmax()

        # Definition of field variable
        u = np.zeros((nr, ntheta + 2))
        # Copy of field variable for output and post-processing
        u_out = np.zeros((nr, ntheta))

        # Initialising Gaussian blob as initial condition of field
        x_center, y_center = config.get_xb_yb()
        x_width, y_width = config.get_wxb_wyb()
        for l in range(mesh.get_n_points_grid()):
            mesh_ind = mesh.grid_to_mesh_index(l)
            x = mesh.get_x(mesh_ind)
            y = mesh.get_y(mesh_ind)
            gaussx = gaussian_blob(x_center, x_width, x)
            gaussy = gaussian_blob(y_center, y_width, y)

            i, j = mesh.get_i_j_from_index(mesh_ind)
            u[i, j] = gaussx * gaussy

        # Setup Dirichlet boundary conditions at inner and outer edge of the torus
        bnd_vals = np.zeros(mesh.get_n_points_ghost())
        set_bnd_vals(mesh, bnd_vals, u)

        # Get parameters from config and mesh modules
        diffc_perp = config.get_diffusion_coeff()
        print("Diffusion coefficient = {}".format(diffc_perp))
        dt = config.get_dt()
        print("dt = {}".format(dt))
        n_t, n_out = config.get_n_timesteps(), config.get_n_output()
        dr = mesh.get_r_spacing()

        # Calculate d_theta (spacing in theta direction)
        dtheta = np.zeros(nr)
        r_c = rmin - dr
        for k in range(nr):
            dtheta[k] = mesh.get_theta_spacing(r_c)
            r_c += dr
        assert (rmax < r_c < rmax + 2 * dr), "theta spacing does not match mesh geometry"

        # Check the CFL Condition for Diffusion Equation
        cfl_r = dt * diffc_perp / (dr * dr)
        print("CFL Coefficient with radial param = {}. Must be less than 0.5".format(cfl_r))
        assert (cfl_r < 0.5)

        cfl_theta = dt * diffc_perp / (rmin * rmin * np.amin(dtheta) * np.mean(dtheta))
        print("CFL Coefficient with theta param = {}. Must be less than 0.5".format(cfl_theta))
        assert (cfl_theta < 0.5)

        # Calculate radius values at each grid point
        r_self, r_minus, r_plus = np.zeros((nr, ntheta + 2)), np.zeros((nr, ntheta + 2)), np.zeros((nr, ntheta + 2))
        for i in range(1, nr - 1):
            for j in range(1, ntheta + 1):
                mesh_ind = mesh.get_index_from_i_j(i, j - 1)
                ind_minus = mesh.get_index_from_i_j(i - 1, j - 1)
                ind_plus = mesh.get_index_from_i_j(i + 1, j - 1)
                # r_(i,j) value
                r_self[i, j] = mesh.get_r(mesh_ind)
                # r_(i-1/2,j) value
                r_minus[i, j] = (mesh.get_r(mesh_ind) + mesh.get_r(ind_minus)) / 2
                # r_(i+1/2,j) value
                r_plus[i, j] = (mesh.get_r(mesh_ind) + mesh.get_r(ind_plus)) / 2

        # Time loop
        for n in range(n_t):
            # Assign values to ghost cells for periodicity in theta direction
            u[:, 0] = u[:, ntheta]
            u[:, ntheta + 1] = u[:, 1]

            # Iterate over all grid points in a Cartesian grid fashion
            for i in range(1, nr - 1):
                for j in range(1, ntheta + 1):
                    # Staggered grid scheme to evaluate derivatives in radial direction
                    du_perp = (r_plus[i, j] * (u[i + 1, j] - u[i, j]) - r_minus[i, j] * (u[i, j] - u[i - 1, j])) / (
                                r_self[i, j] * dr * dr)

                    # Second order central difference components in theta direction
                    du_perp += (u[i, j - 1] + u[i, j + 1] - 2 * u[i, j]) / (r_self[i, j] * r_self[i, j] * dtheta[i] * dtheta[i])

                    u[i, j] += du_perp * dt * diffc_perp
                    u_out[i, j - 1] = u[i, j]

            if n % n_out == 0:
                write_vtk(u_out, mesh, n)
                print("Elapsed time = {}  || Field sum = {}".format(n * dt, u_out.sum()))
                print("Elapsed CPU time = {}".format(time.clock()))

        print("CPU time = {}".format(time.clock()))
        # End
