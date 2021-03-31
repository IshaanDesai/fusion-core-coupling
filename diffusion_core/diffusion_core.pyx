"""
Code to simulate diffusion in a polar coordinate system replicating a gyrokinetics fusion code (reactor core physics).
Verification with analytical solution obtained using Bessel functions of the first kind
"""

import numpy as np
cimport numpy as np
cimport cython
from diffusion_core.modules.mesh_2d import Mesh, MeshVertexType
from diffusion_core.modules.output import write_vtk, write_csv
from diffusion_core.modules.config import Config
from diffusion_core.modules.boundary import Boundary, BoundaryType
from diffusion_core.modules.ansol import Ansol
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
        mesh = Mesh(config)
        nr, ntheta = config.get_r_points(), config.get_theta_points()
        rmin, rmax = config.get_rmin(), config.get_rmax()

        # Create MMS module object
        ansol_bessel = Ansol()

        # Field variable array
        u_np = np.zeros((nr, ntheta), dtype=np.double)
        cdef double [:, ::1] u = u_np
        # Field delta change array
        du_perp_np = np.zeros((nr, ntheta), dtype=np.double)
        cdef double [:, ::1] du_perp = du_perp_np

        # Setting initial state of the field using analytical solution formulation
        for l in range(mesh.get_n_points_grid()):
            mesh_ind = mesh.grid_to_mesh_index(l)
            radialp = mesh.get_r(mesh_ind)
            thetap = mesh.get_theta(mesh_ind)

            i, j = mesh.get_i_j_from_index(mesh_ind)
            u[i, j] = ansol_bessel.ansol(radialp, thetap, 0)

        # Initialize boundary conditions at inner and outer edge of the torus
        bndvals_wall = np.zeros((mesh.get_n_points_wall()))
        bnd_wall = Boundary(config, mesh, bndvals_wall, u, BoundaryType.DIRICHLET, MeshVertexType.BC_WALL)
        bndvals_core = np.zeros((mesh.get_n_points_core()))
        bnd_core = Boundary(config, mesh, bndvals_core, u, BoundaryType.DIRICHLET, MeshVertexType.BC_CORE)

        # Reset boundary conditions according to analytical solution
        bnd_wall.set_bnd_vals_ansol(ansol_bessel, u, 0)
        bnd_core.set_bnd_vals_ansol(ansol_bessel, u, 0)

        # Get parameters from config and mesh modules
        diffc_perp = config.get_diffusion_coeff()
        self.logger.info('Diffusion coefficient = %f', diffc_perp)
        cdef double dt = config.get_dt()
        self.logger.info('dt = %f', dt)
        t_total, t_out = config.get_total_time(), config.get_t_output()
        cdef int n_t = int(t_total/dt)
        cdef int n_out = int(t_out/dt)

        cdef double dr = mesh.get_r_spacing()
        cdef double dtheta = mesh.get_theta_spacing()

        # Calculate radius and theta values at each grid point
        r_self_np = np.zeros((nr, ntheta), dtype=np.double)
        cdef double [:, ::1] r_self = r_self_np
        r_minus_np = np.zeros((nr, ntheta), dtype=np.double)
        cdef double [:, ::1] r_minus = r_minus_np
        r_plus_np = np.zeros((nr, ntheta), dtype=np.double)
        cdef double [:, ::1] r_plus = r_plus_np

        theta_self_np = np.zeros((nr, ntheta), dtype=np.double)
        cdef double [:, ::1] theta_self = theta_self_np

        for i in range(1, nr - 1):
            for j in range(ntheta):
                mesh_ind = mesh.get_index_from_i_j(i, j)
                ind_minus = mesh.get_index_from_i_j(i - 1, j)
                ind_plus = mesh.get_index_from_i_j(i + 1, j)
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
            for i in range(1, nr - 1):
                # Calculating for points theta = 0
                # Staggered grid scheme to evaluate derivatives in radial direction
                du_perp[i, 0] = (r_plus[i, 0]*(u[i+1, 0] - u[i, 0]) - r_minus[i, 0]*(u[i, 0] - u[i-1, 0])) / (
                    r_self[i, 0]*dr*dr)
                # Second order central difference components in theta direction
                du_perp[i, 0] += (u[i, ntheta-1] + u[i, 1] - 2*u[i, 0]) / (r_self[i, 0]*r_self[i, 0]*dtheta*dtheta)

                # Calculating for points theta = 2*pi - dtheta
                # Staggered grid scheme to evaluate derivatives in radial direction
                du_perp[i, ntheta-1] = (r_plus[i, ntheta-1]*(u[i+1, ntheta-1] - u[i, ntheta-1]) -
                    r_minus[i, ntheta-1]*(u[i, ntheta-1] - u[i-1, ntheta-1])) / (r_self[i, ntheta-1]*dr*dr)
                # Second order central difference components in theta direction
                du_perp[i, ntheta-1] += (u[i, ntheta-2] + u[i, 0] - 2*u[i, ntheta-1]) / (
                    r_self[i, ntheta-1]*r_self[i, ntheta-1]*dtheta*dtheta)

            # Iterate over all grid points in a Cartesian grid fashion
            for i in range(1, nr - 1):
                for j in range(1, ntheta - 1):
                    # Staggered grid scheme to evaluate derivatives in radial direction
                    du_perp[i, j] = (r_plus[i, j]*(u[i+1, j] - u[i, j]) - r_minus[i, j]*(u[i, j] - u[i-1, j])) / (
                               r_self[i, j]*dr*dr)

                    # Second order central difference components in theta direction
                    du_perp[i, j] += (u[i, j-1] + u[i, j+1] - 2*u[i, j]) / (r_self[i, j]*r_self[i, j]*dtheta*dtheta)

            # Update scheme
            for i in range(1, nr - 1):
                for j in range(ntheta):
                    u[i, j] += dt*diffc_perp*du_perp[i, j]

            # Set Neumann boundary conditions in each iteration
            bnd_wall.set_bnd_vals_ansol(ansol_bessel, u, n*dt)
            bnd_core.set_bnd_vals_ansol(ansol_bessel, u, n*dt)

            if n%n_out == 0 or n == n_t-1:
                write_csv(u, mesh, n+1)
                write_vtk(u, mesh, n+1)
                self.logger.info('VTK file output written at t = %f', n*dt)
                u_sum = 0
                for i in range(nr):
                    for j in range(0, ntheta):
                        u_sum += u[i, j]

                self.logger.info("Elapsed time = {}  || Field sum = {}".format(n*dt, u_sum/(nr*ntheta)))
                self.logger.info("Elapsed CPU time = {}".format(time.process_time()))

        ansol_bessel.compare_ansoln(mesh, u, n*dt)

        self.logger.info("Total CPU time = {}".format(time.process_time()))
        # End
