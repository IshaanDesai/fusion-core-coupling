"""
Code to simulate diffusion in a polar coordinate system replicating a gyrokinetics fusion code (reactor core physics)
"""

import numpy as np
cimport numpy as np
cimport cython
from diffusion_core.modules.mesh_2d import Mesh, MeshVertexType
from diffusion_core.modules.output import write_vtk
from diffusion_core.modules.config import Config
from diffusion_core.modules.boundary import set_bnd_vals
from diffusion_core.modules.initialization import gaussian_blob
from diffusion_core.modules.mms cimport MMS
import math
import time


class Diffusion:
    def __init__(self):
        self._file = None

    def solve_diffusion(self):
        # Read initial conditions from a JSON config file
        problem_config_file = "diffusion-coupling-config.json"
        config = Config(problem_config_file)

        # Iterators
        cdef Py_ssize_t i, j

        # Mesh setup
        mesh = Mesh(problem_config_file)
        nr, ntheta = mesh.get_n_points_axiswise()
        rmin, rmax = config.get_rmin(), config.get_rmax()

        # Create MMS module object
        mms = MMS(config, mesh)

        # Definition of field variable
        u_numpy = np.zeros((nr, ntheta + 2), dtype=np.double)
        cdef double [:, ::1] u = u_numpy

        # Initializing Gaussian blob as initial condition of field
        # x_center, y_center = config.get_xb_yb()
        # x_width, y_width = config.get_wxb_wyb()
        # for l in range(mesh.get_n_points_grid()):
        #     mesh_ind = mesh.grid_to_mesh_index(l)
        #     x = mesh.get_x(mesh_ind)
        #     y = mesh.get_y(mesh_ind)
        #     gaussx = gaussian_blob(x_center, x_width, x)
        #     gaussy = gaussian_blob(y_center, y_width, y)

        #     i, j = mesh.get_i_j_from_index(mesh_ind)
        #     u[i, j] = gaussx * gaussy

        # Initializing custom initial state for MMS analysis
        for l in range(mesh.get_n_points_grid()):
            mesh_ind = mesh.grid_to_mesh_index(l)
            radialp = mesh.get_r(mesh_ind)
            thetap = mesh.get_theta(mesh_ind)

            i, j = mesh.get_i_j_from_index(mesh_ind)
            u[i, j] = mms.init_mms(radialp, thetap)

        # Setup Dirichlet boundary conditions at inner and outer edge of the torus
        bnd_vals = np.zeros(mesh.get_n_points_ghost())
        set_bnd_vals(mesh, bnd_vals, u)

        # Get parameters from config and mesh modules
        diffc_perp = config.get_diffusion_coeff()
        print("Diffusion coefficient = {}".format(diffc_perp))
        cdef double dt = config.get_dt()
        print("dt = {}".format(dt))
        n_t, n_out = config.get_n_timesteps(), config.get_n_output()

        cdef double dr = mesh.get_r_spacing()
        cdef double dtheta = 2 * math.pi / config.get_theta_points()

        # Calculate radius and theta values at each grid point
        r_self_numpy = np.zeros((nr, ntheta + 2), dtype=np.double)
        cdef double [:, ::1] r_self = r_self_numpy
        r_minus_numpy = np.zeros((nr, ntheta + 2), dtype=np.double)
        cdef double [:, ::1] r_minus = r_minus_numpy
        r_plus_numpy = np.zeros((nr, ntheta + 2), dtype=np.double)
        cdef double [:, ::1] r_plus = r_plus_numpy

        theta_self_numpy = np.zeros((nr, ntheta + 2), dtype=np.double)
        cdef double [:, ::1] theta_self = theta_self_numpy

        for i in range(1, nr - 1):
            for j in range(1, ntheta + 1):
                mesh_ind = mesh.get_index_from_i_j(i, j - 1)
                ind_minus = mesh.get_index_from_i_j(i - 1, j - 1)
                ind_plus = mesh.get_index_from_i_j(i + 1, j - 1)
                # r_(i,j) value
                r_self[i, j] = mesh.get_r(mesh_ind)
                # r_(i-1/2,j) value
                r_minus[i, j] = (mesh.get_r(mesh_ind) + mesh.get_r(ind_minus)) / 2.0
                # r_(i+1/2,j) value
                r_plus[i, j] = (mesh.get_r(mesh_ind) + mesh.get_r(ind_plus)) / 2.0
                # theta_(i,j) value
                theta_self[i, j] = mesh.get_theta(mesh_ind)

        # Check the CFL Condition for Diffusion Equation
        cfl_r = dt * diffc_perp / (dr * dr)
        print("CFL Coefficient with radial param = {}. Must be less than 0.5".format(cfl_r))
        cfl_theta = dt * diffc_perp / (np.mean(r_self) * np.mean(r_self) * dtheta * dtheta)
        print("CFL Coefficient with theta param = {}. Must be less than 0.5".format(cfl_theta))
        assert (cfl_r < 0.5)
        assert (cfl_theta < 0.5)

        # Time loop
        cdef double du_perp, u_sum
        for n in range(n_t):
            # Assign values to ghost cells for periodicity in theta direction
            for i in range(nr):
                u[i, 0] = u[i, ntheta]
                u[i, ntheta + 1] = u[i, 1]

            # Iterate over all grid points in a Cartesian grid fashion
            for i in range(1, nr - 1):
                for j in range(1, ntheta + 1):
                    du_perp = 0.0

                    # Staggered grid scheme to evaluate derivatives in radial direction
                    du_perp += (r_plus[i, j] * (u[i + 1, j] - u[i, j]) - r_minus[i, j] * (u[i, j] - u[i - 1, j])) / (
                               r_self[i, j] * dr * dr)

                    # Second order central difference components in theta direction
                    du_perp += (u[i, j - 1] + u[i, j + 1] - 2 * u[i, j]) / (r_self[i, j] * r_self[i, j] * dtheta * dtheta)

                    # Adding pseudo source term for MMS
                    u[i, j] += du_perp * dt * diffc_perp + dt * mms.source_term(r_self[i, j], theta_self[i, j], n*dt)

            if n % n_out == 0:
                write_vtk(u, mesh, n)
                u_sum = 0
                for i in range(nr):
                    for j in range(1, ntheta + 1):
                        u_sum += u[i, j]

                print("Elapsed time = {}  || Field sum = {}".format(n * dt, u_sum/(nr*ntheta)))
                print("Elapsed CPU time = {}".format(time.clock()))
                # Output L2 error for MMS
                print("dr = {}, dtheta = {}, dt = {} and at t = {} | L2 error = {}".format(dr, dtheta, dt, n*dt, mms.error_computation(mesh, u, n*dt)))

        print("Total CPU time = {}".format(time.clock()))
        # End
