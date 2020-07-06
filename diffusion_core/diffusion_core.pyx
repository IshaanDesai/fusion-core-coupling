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
import logging

class Diffusion:
    def __init__(self):
        self.logger = logging.getLogger('main.diffusion_core.Diffusion')
        self._file = None

    def solve_diffusion(self):
        self.logger.info('Solving Diffusion case')
        # Read initial conditions from a JSON config file
        problem_config_file = "diffusion-coupling-config.json"
        config = Config(problem_config_file)

        # Iterators
        cdef Py_ssize_t i, j

        # Mesh setup
        mesh = Mesh(problem_config_file)
        nr, ntheta = config.get_r_points(), config.get_theta_points()
        rmin, rmax = config.get_rmin(), config.get_rmax()

        # Create MMS module object
        mms = MMS(config, mesh)

        # Definition of field variable
        u_numpy = np.zeros((nr, ntheta + 2), dtype=np.double)
        cdef double [:, ::1] u = u_numpy
        du_perp_numpy = np.zeros((nr, ntheta + 2), dtype=np.double)
        cdef double [:, ::1] du_perp = du_perp_numpy

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
        self.logger.info('Diffusion coefficient = %f', diffc_perp)
        cdef double dt = config.get_dt()
        self.logger.info('dt = %f', dt)
        t_total, t_out = config.get_total_time(), config.get_t_output()
        cdef int n_t = int(t_total/dt)
        cdef int n_out = int(t_out/dt)

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
        self.logger.info('CFL Coefficient with radial param = %f. Must be less than 0.5', cfl_r)
        cfl_theta = dt * diffc_perp / (np.mean(r_self) * np.mean(r_self) * dtheta * dtheta)
        self.logger.info('CFL Coefficient with theta param = %f. Must be less than 0.5', cfl_theta)
        assert (cfl_r < 0.5)
        assert (cfl_theta < 0.5)

        # Time loop
        cdef double u_sum
        for n in range(n_t):
            # Assign values to ghost cells for periodicity in theta direction
            for i in range(nr):
                u[i, 0] = u[i, ntheta]
                u[i, ntheta + 1] = u[i, 1]

            # Iterate over all grid points in a Cartesian grid fashion
            for i in range(1, nr - 1):
                for j in range(1, ntheta + 1):
                    # Staggered grid scheme to evaluate derivatives in radial direction
                    du_perp[i, j] = (r_plus[i, j]*(u[i + 1, j] - u[i, j]) - r_minus[i, j]*(u[i, j] - u[i - 1, j])) / (
                               r_self[i, j]*dr*dr)

                    # Second order central difference components in theta direction
                    du_perp[i, j] += (u[i, j - 1] + u[i, j + 1] - 2*u[i, j]) / (r_self[i, j]*r_self[i, j]*dtheta*dtheta)

            # Update scheme
            for i in range(1, nr - 1):
                for j in range(1, ntheta + 1):
                    u[i, j] += dt*diffc_perp*du_perp[i, j] + dt*mms.source_term(r_self[i, j], theta_self[i, j], n*dt)

            if n%n_out==0 or n==n_t-1:
                write_vtk(u, mesh, n)
                self.logger.info('VTK file output written at t = %f', n*dt)
                u_sum = 0
                for i in range(nr):
                    for j in range(1, ntheta + 1):
                        u_sum += u[i, j]

                self.logger.info('Elapsed time = %f  || Field sum = %f', n*dt, u_sum/(nr*ntheta))
                self.logger.info('Elapsed CPU time = %f', time.clock())
                # Output L2 error for MMS
                self.logger.info('dr = %f, dtheta = %f, dt = %f and at t = %f | L2 error = %f', dr, dtheta, dt, n*dt, mms.error_computation(mesh, u, n*dt))

        self.logger.info('Total CPU time = %f', time.clock())
        # End
