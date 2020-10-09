"""
Code to simulate diffusion in a polar coordinate system replicating a gyrokinetics fusion code (reactor core physics)
"""

import numpy as np
cimport numpy as np
cimport cython
from diffusion_core.modules.mesh_2d import Mesh, MeshVertexType
from diffusion_core.modules.output import write_vtk, write_csv
from diffusion_core.modules.config import Config
from diffusion_core.modules.boundary import Boundary, BoundaryType
from diffusion_core.modules.mms cimport MMS
import math
import time
import logging
import netCDF4 as nc

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

        # Read metric coefficient NetCDF file
        ds = nc.Dataset('./polars.nc')

        # Mesh setup
        nrho, ntheta = ds.dimensions['nrho'], ds.dimensions['ntheta']
        assert nrho == config.get_r_points()
        assert ntheta == config.get_theta_points()

        # Get geometrical variables from NetCDF file
        rho_np, theta_np = np.array(ds['rho'][:]), np.array(ds['theta'][:])
        cdef double [:, ::1] rho = rho_np
        cdef double [:, ::1] theta = theta_np
        xpol_np, ypol_np = np.array(ds['xpol'][:]), np.array(ds['ypol'][:])
        cdef double [:, ::1] xpol = xpol_np
        cdef double [:, ::1] ypol = ypol_np

        # Get Jacobian from NetCDF file
        jac_np = np.array(ds['jacobian'][:])
        cdef double [:, ::1] jac = jac_np

        # Get metric coefficients from NetCDF file
        g_rhorho_np, g_rhotheta_np = np.array(ds['g_rhorho'][:]), np.array(ds['g_rhotheta'][:])
        cdef double [:, ::1] g_rhorho = g_rhorho_np
        cdef double [:, ::1] g_rhotheta = g_rhotheta_np
        g_thetatheta_np, g_phiphi_np = np.array(ds['g_thetatheta'][:]), np.array(ds['g_phiphi'][:])
        cdef double [:, ::1]  g_thetatheta = g_thetatheta_np
        cdef double [:, ::1] g_phiphi = g_phiphi_np

        # Field variable array
        u_np = np.zeros((nrho, ntheta), dtype=np.double)
        cdef double [:, ::1] u = u_np
        # Field delta change array
        du_perp_np = np.zeros((nrho, ntheta), dtype=np.double)
        cdef double [:, ::1] du_perp = du_perp_np

        # Initializing custom initial state (sinosoidal)
        for i in range(nrho):
            for j in range(ntheta):
                u[i, j] = mms.init_mms(rho[i], theta[j])

        # Set boundary conditions (Dirichlet)
        for i in range(nrho):
            for j in range(ntheta):
                if i == 0 or i == nrho-1:
                    u[i, j] = 0.0

        # Get parameters from config and mesh modules
        diffc_perp = config.get_diffusion_coeff()
        self.logger.info('Diffusion coefficient = %f', diffc_perp)
        cdef double dt = config.get_dt()
        self.logger.info('dt = %f', dt)
        t_total, t_out = config.get_total_time(), config.get_t_output()
        cdef int n_t = int(t_total/dt)
        cdef int n_out = int(t_out/dt)

        cdef double drho = ds.globalattributes['drho']
        cdef double dtheta = ds.globalattributes['dtheta']

        # Calculate rho and theta values at each grid point
        rho_minus_np = np.zeros((nrho, ntheta), dtype=np.double)
        cdef double [:, ::1] rho_minus = rho_minus_np
        rho_plus_np = np.zeros((nrho, ntheta), dtype=np.double)
        cdef double [:, ::1] rho_plus = rho_plus_np

        for i in range(1, nrho - 1):
            for j in range(ntheta):
                # rho_(i,j) value
                rho_self[i, j] = rho[i, j]
                # rho_(i-1/2,j) value
                rho_minus[i, j] = (rho[i-1, j] + rho[i, j]) / 2
                # r_(i+1/2,j) value
                r_plus[i, j] = (rho[i+1, j] + rho[i, j]) / 2

        # Check the CFL Condition for Diffusion Equation
        cfl_rho = dt * diffc_perp / (drho * drho)
        self.logger.info('CFL Coefficient with radial param = %f. Must be less than 0.5', cfl_rho)
        cfl_theta = dt * diffc_perp / (np.mean(rho) * np.mean(rho) * dtheta * dtheta)
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
                du_perp[i, 0] = (rho_plus[i, 0]*(u[i+1, 0] - u[i, 0]) - rho_minus[i, 0]*(u[i, 0] - u[i-1, 0])) / (
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
            bnd_wall.set_bnd_vals_mms(u, n*dt)
            bnd_core.set_bnd_vals_mms(u, n*dt)

            if n%n_out == 0 or n == n_t-1:
                write_csv(u, mesh, n+1)
                write_vtk(u, mesh, n+1)
                self.logger.info('VTK file output written at t = %f', n*dt)
                u_sum = 0
                for i in range(nr):
                    for j in range(0, ntheta):
                        u_sum += u[i, j]

                self.logger.info("Elapsed time = {}  || Field sum = {}".format(n*dt, u_sum/(nr*ntheta)))
                self.logger.info("Elapsed CPU time = {}".format(time.clock()))

        self.logger.info("Total CPU time = {}".format(time.clock()))
        # End
