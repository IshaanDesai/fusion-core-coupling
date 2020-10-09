"""
Code to simulate diffusion in a polar coordinate system replicating a gyrokinetics fusion code (reactor core physics)
"""

import numpy as np
cimport numpy as np
cimport cython
from diffusion_core.modules.output import write_vtk, write_csv
from diffusion_core.modules.config import Config
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
        g_rr_np, g_rt_np = np.array(ds['g_rhorho'][:]), np.array(ds['g_rhotheta'][:])
        cdef double [:, ::1] g_rr = g_rr_np
        cdef double [:, ::1] g_rt = g_rt_np
        g_tt_np, = np.array(ds['g_thetatheta'][:])
        cdef double [:, ::1]  g_tt = g_tt_np

        # Field variable array
        u_np = np.zeros((nrho, ntheta), dtype=np.double)
        cdef double [:, ::1] u = u_np
        # Field delta change array
        du_np = np.zeros((nrho, ntheta), dtype=np.double)
        cdef double [:, ::1] du = du_np

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
            jp = [ntheta-2, ntheta-1, 0, 1]
            for i in range(1, nrho - 1):
                for j in range(1, 3):
                    # Staggered grid for rho-rho diagonal term
                    du[i, jp[j]] = ((jac[i+1, jp[j]] + jac[i, jp[j]])*(g_rr[i+1, jp[j]] + g_rr[i, jp[j]])*(u[i+1, jp[j]] - u[i, jp[j]]) -
                        (jac[i, jp[j]] + jac[i-1, jp[j]])*(g_rr[i, jp[j]] + g_rr[i-1, jp[j]])*(u[i, jp[j]] - u[i-1, jp[j]])) / (4*drho*drho)

                    # Staggered grid for theta-theta diagonal term
                    du[i, jp[j]] += ((jac[i, jp[j+1]] + jac[i, jp[j]])*(g_tt[i, jp[j+1]] + g_tt[i, jp[j]])*(u[i, jp[j+1]] - u[i, jp[j]]) -
                        (jac[i, jp[j]] + jac[i, jp[j-1]])*(g_tt[i, jp[j]] + g_tt[i, jp[j-1]])*(u[i, jp[j]] - u[i-1, jp[j]])) / (4*dtheta*dtheta)

                    # Off-diagonal term rho-theta
                    du[i, jp[j]] += (jac[i+1, jp[j]]*g_rt[i+1, jp[j]]*(u[i+1, jp[j+1]] - u[i+1, jp[j-1]]) -
                        jac[i-1, jp[j]]*g_rt[i-1, jp[j]]*(u[i-1, jp[j+1]] - u[i-1, jp[j-1]])) / (4*drho*dtheta)

                    # Off-diagonal term theta-rho
                    du[i, jp[j]] += (jac[i, jp[j+1]]*g_rt[i, jp[j+1]]*(u[i+1, jp[j+1]] - u[i-1, jp[j+1]]) -
                        jac[i, jp[j-1]]*g_rt[i, jp[j-1]]*(u[i+1, jp[j-1]] - u[i-1, jp[j-1])) / (4*dtheta*drho)

            # Iterate over all grid points in a Cartesian grid fashion
            for i in range(1, nrho-1):
                for j in range(1, ntheta-1):
                    # Staggered grid for rho-rho diagonal term
                    du[i, j] = ((jac[i+1, j] + jac[i, j])*(g_rr[i+1, j] + g_rr[i, j])*(u[i+1, j] - u[i, j]) -
                        (jac[i, j] + jac[i-1, j])*(g_rr[i, j] + g_rr[i-1,j])*(u[i, j] - u[i-1, j])) / (4*drho*drho)

                    # Staggered grid for theta-theta diagonal term
                    du[i, j] += ((jac[i, j+1] + jac[i, j])*(g_tt[i, j+1] + g_tt[i, j])*(u[i, j+1] - u[i, j]) -
                        (jac[i, j] + jac[i, j-1])*(g_tt[i, j] + g_tt[i, j-1])*(u[i, j] - u[i-1, j])) / (4*dtheta*dtheta)

                    # Off-diagonal term rho-theta
                    du[i, j] += (jac[i+1, j]*g_rt[i+1, j]*(u[i+1, j+1] - u[i+1, j-1]) -
                        jac[i-1, j]*g_rt[i-1, j]*(u[i-1, j+1] - u[i-1, j-1])) / (4*drho*dtheta)

                    # Off-diagonal term theta-rho
                    du[i, j] += (jac[i, j+1]*g_rt[i, j+1]*(u[i+1, j+1] - u[i-1, j+1]) -
                        jac[i, j-1]*g_rt[i, j-1]*(u[i+1, j-1] - u[i-1, j-1])) / (4*dtheta*drho)

            # Update scheme
            for i in range(1, nr - 1):
                for j in range(ntheta):
                    u[i, j] += dt*diffc*du[i, j] / jac[i, j]

            if n%n_out == 0 or n == n_t-1:
                write_csv(u, xpol, ypol, n+1)
                write_vtk(u, xpol, ypol n+1)
                self.logger.info('VTK file output written at t = %f', n*dt)
                u_sum = 0
                for i in range(nrho):
                    for j in range(0, ntheta):
                        u_sum += u[i, j]

                self.logger.info("Elapsed time = {}  || Field sum = {}".format(n*dt, u_sum/(nrho*ntheta)))
                self.logger.info("Elapsed CPU time = {}".format(time.clock()))

        self.logger.info("Total CPU time = {}".format(time.clock()))
        # End
