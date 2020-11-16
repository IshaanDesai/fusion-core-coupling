"""
Code to simulate diffusion in a polar coordinate system replicating a gyrokinetics fusion code (reactor core physics)
"""

import numpy as np
cimport numpy as np
cimport cython
from diffusion_core.modules.output import write_vtk, write_csv, write_custom_csv
from diffusion_core.modules.config import Config
import math
import time
import logging
import netCDF4 as nc
import precice

class Diffusion:
    def __init__(self):
        self.logger = logging.getLogger('main.diffusion_core.Diffusion')
        self._file = None

        # Read initial conditions from a JSON config file
        self._config = Config('diffusion-coupling-config.json')

        # Define coupling interface
        self._interface = precice.Interface(self._config.get_participant_name(), self._config.get_config_file_name(), 0, 1)

        # Coupling mesh
        self._coupling_write_mesh_vertices = None
        self._vertex_ids = None

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
        nrho, ntheta = ds.dimensions['nrho'].size, ds.dimensions['ntheta'].size
        assert nrho == config.get_rho_points()
        assert ntheta == config.get_theta_points()

        # Get geometrical variables from NetCDF file
        rho_np, theta_np = np.array(ds['rho'][:]), np.array(ds['theta'][:])
        cdef double [::1] rho = rho_np
        cdef double [::1] theta = theta_np
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
        g_tt_np = np.array(ds['g_thetatheta'][:])
        cdef double [:, ::1]  g_tt = g_tt_np

        # Field variable array
        u_np = np.zeros((ntheta, nrho), dtype=np.double)
        cdef double [:, ::1] u = u_np
        # Field delta change array
        du_np = np.zeros((ntheta, nrho), dtype=np.double)
        cdef double [:, ::1] du = du_np

        # Initializing gaussian blob
        xb, yb = config.get_xb_yb()
        wxb, wyb = config.get_wxb_wyb()
        for i in range(ntheta):
            for j in range(nrho):
                x = xpol[i, j]
                y = ypol[i, j]
                gaussx = math.exp(-(x-xb)*(x-xb) / (wxb*wxb))
                gaussy = math.exp(-(y-yb)*(y-yb) / (wyb*wyb))
                u[i, j] = gaussx*gaussy

        # Set boundary conditions for inner edge (Dirichlet Zero)
        for i in range(ntheta):
            for j in range(nrho):
                if j == 0:
                    u[i, j] = 0.0

        # Set boundary conditions for outer edge (Neumann Zero) # THIS IMPLEMENTATION IS WRONG
        for i in range(ntheta):
            for j in range(nrho):
                if j == nrho-1:
                    u[i, j] = u[i, j-1]

        # Setup coupling mesh
        vertices = []
        for i in range(ntheta):
            for j in range(nrho):
                if j == nrho-1:
                    vertices.append([xpol[i, j], ypol[i, j]])

        self._coupling_write_mesh_vertices = np.array(vertices)
        write_custom_csv(vertices)

        self._write_vertex_ids = self._interface.set_mesh_vertices(self._interface.get_mesh_id(
            self._config.get_coupling_mesh_name()), self._coupling_write_mesh_vertices)

        # Set up write mesh in preCICE
        self._write_mesh_id = self._interface.get_mesh_id(self._config.get_coupling_mesh_name())
        self._write_data_id = self._interface.get_data_id(self._config.get_write_data_name(), self._write_mesh_id)

        # Get parameters from config and mesh modules
        diffc = config.get_diffusion_coeff()
        self.logger.info('Diffusion coefficient = %f', diffc)
        cdef double dt = config.get_dt()
        self.logger.info('dt = %f', dt)
        t_total, t_out = config.get_total_time(), config.get_t_output()

        cdef double drho = ds.getncattr('drho')
        cdef double dtheta = ds.getncattr('dtheta')

        # Check the CFL Condition for Diffusion Equation
        cfl_rho = dt * diffc / (drho * drho)
        self.logger.info('CFL Coefficient with radial param = %f. Must be less than 0.5', cfl_rho)
        cfl_theta = dt * diffc / (np.mean(rho) * np.mean(rho) * dtheta * dtheta)
        self.logger.info('CFL Coefficient with theta param = %f. Must be less than 0.5', cfl_theta)
        assert (cfl_rho < 0.5)
        assert (cfl_theta < 0.5)

        # Initialize preCICE interface
        cdef double precice_dt = self._interface.initialize()

        dt = min(precice_dt, dt)
        cdef int n_t = int(t_total/dt)
        cdef int n_out = int(t_out/dt)
        print("n_t = {}, n_out = {}".format(n_t, n_out))

        # Write initial state
        write_csv(u, xpol, ypol, 0)
        write_vtk(u, xpol, ypol, 0)

        # Time loop
        cdef double u_sum
        # Time loop
        cdef int n = 0
        cdef double t = 0.0
        while self._interface.is_coupling_ongoing():
            # Update time step
            dt = min(precice_dt, dt)

            # Assign values to ghost cells for periodicity in theta direction
            ip = [ntheta-2, ntheta-1, 0, 1]
            for i in range(1, 3):
                for j in range(1, nrho-1):
                    # Staggered grid for rho-rho diagonal term
                    du[ip[i], j] = ((jac[ip[i], j+1] + jac[ip[i], j])*(g_rr[ip[i], j+1] + g_rr[ip[i], j])*(u[ip[i], j+1] - u[ip[i], j]) -
                        (jac[ip[i], j] + jac[ip[i], j-1])*(g_rr[ip[i], j] + g_rr[ip[i], j-1])*(u[ip[i], j] - u[ip[i], j-1])) / (4*drho*drho)

                    # Staggered grid for theta-theta diagonal term
                    du[ip[i], j] += ((jac[ip[i+1], j] + jac[ip[i], j])*(g_tt[ip[i+1], j] + g_tt[ip[i], j])*(u[ip[i+1], j] - u[ip[i], j]) -
                        (jac[ip[i], j] + jac[ip[i-1], j])*(g_tt[ip[i], j] + g_tt[ip[i-1], j])*(u[ip[i], j] - u[ip[i-1], j])) / (4*dtheta*dtheta)

                    # Off-diagonal term rho-theta
                    du[ip[i], j] += (jac[ip[i], j+1]*g_rt[ip[i], j+1]*(u[ip[i+1], j+1] - u[ip[i-1], j+1]) -
                        jac[ip[i], j-1]*g_rt[ip[i], j-1]*(u[ip[i+1], j-1] - u[ip[i-1], j-1])) / (4*drho*dtheta)

                    # Off-diagonal term theta-rho
                    du[ip[i], j] += (jac[ip[i+1], j]*g_rt[ip[i+1], j]*(u[ip[i+1], j+1] - u[ip[i+1], j-1]) -
                        jac[ip[i-1], j]*g_rt[ip[i-1], j]*(u[ip[i-1], j+1] - u[ip[i-1], j-1])) / (4*dtheta*drho)

            # Iterate over all grid points
            for i in range(1, ntheta-1):
                for j in range(1, nrho-1):
                    # Staggered grid for rho-rho diagonal term
                    du[i, j] = ((jac[i, j+1] + jac[i, j])*(g_rr[i, j+1] + g_rr[i, j])*(u[i, j+1] - u[i, j]) -
                        (jac[i, j] + jac[i, j-1])*(g_rr[i, j] + g_rr[i, j-1])*(u[i, j] - u[i, j-1])) / (4*drho*drho)

                    # Staggered grid for theta-theta diagonal term
                    du[i, j] += ((jac[i+1, j] + jac[i, j])*(g_tt[i+1, j] + g_tt[i, j])*(u[i+1, j] - u[i, j]) -
                        (jac[i, j] + jac[i-1, j])*(g_tt[i, j] + g_tt[i-1, j])*(u[i, j] - u[i-1, j])) / (4*dtheta*dtheta)

                    # Off-diagonal term rho-theta
                    du[i, j] += (jac[i, j+1]*g_rt[i, j+1]*(u[i+1, j+1] - u[i-1, j+1]) -
                        jac[i, j-1]*g_rt[i, j-1]*(u[i+1, j-1] - u[i-1, j-1])) / (4*drho*dtheta)

                    # Off-diagonal term theta-rho
                    du[i, j] += (jac[i+1, j]*g_rt[i+1, j]*(u[i+1, j+1] - u[i+1, j-1]) -
                        jac[i-1, j]*g_rt[i-1, j]*(u[i-1, j+1] - u[i-1, j-1])) / (4*dtheta*drho)

            # Update scheme
            for i in range(ntheta):
                for j in range(nrho):
                    u[i, j] += dt*diffc*du[i, j] / jac[i, j]

            # Update boundary conditions for outer edge (Neumann Zero) # THIS IMPLEMENTATION IS WRONG
            for i in range(ntheta):
                for j in range(nrho):
                    if j == nrho-1:
                        u[i, j] = u[i, j-1]

            # Write data to coupling interface preCICE
            scalar_vals = []
            for i in range(ntheta):
                for j in range(nrho):
                    if j == nrho-1:
                        scalar_vals.append(u[i, j])
            self._interface.write_block_scalar_data(self._write_data_id, self._write_vertex_ids, np.array(scalar_vals))

            # Advance coupling via preCICE
            precice_dt = self._interface.advance(dt)

            # update solution
            n += 1
            t += dt

            if n%n_out == 0 or n == n_t-1:
                write_csv(u, xpol, ypol, n+1)
                write_vtk(u, xpol, ypol, n+1)
                self.logger.info('VTK file output written at t = %f', n*dt)
                u_sum = 0
                for i in range(ntheta):
                    for j in range(nrho):
                        u_sum += u[i, j]

                self.logger.info("Elapsed time = {}  || Field sum = {}".format(n*dt, u_sum/(nrho*ntheta)))
                self.logger.info("Elapsed CPU time = {}".format(time.clock()))

        self._interface.finalize()
        self.logger.info("Total CPU time = {}".format(time.clock()))
        # End
