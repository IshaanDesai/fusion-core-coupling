"""
Code to simulate diffusion in a polar coordinate system replicating a gyrokinetics fusion code (reactor core physics).
Verification with analytical solution obtained using Bessel functions of the first kind
"""

import numpy as np
cimport numpy as np
cimport cython
from diffusion_core.modules.output import write_vtk, write_csv, write_custom_csv
from diffusion_core.modules.config import Config
from diffusion_core.modules.boundary import Boundary
from diffusion_core.modules.ansol import Ansol
from diffusion_core.modules.mesh import Mesh
import math
from time import process_time
import logging
import netCDF4 as nc
import precice

class Diffusion:
    def __init__(self):
        self.logger = logging.getLogger('main.diffusion_core.Diffusion')

        # Read initial conditions from a JSON config file
        self._config = Config('diffusion-coupling-config.json')

        # Define coupling interface
        self._interface = precice.Interface(self._config.get_participant_name(), self._config.get_config_file_name(), 0, 1)

        # Coupling mesh
        self._coupling_write_mesh_vertices = None
        self._vertex_ids = None

    def solve_diffusion(self):
        start_time = process_time()

        self.logger.info('Solving Diffusion case')
        # Read initial conditions from a JSON config file
        problem_config_file = "diffusion-coupling-config.json"
        config = Config(problem_config_file)

        # Check if the case is coupling or not
        coupling_on = self._config.is_coupling_ongoing()

        # Iterators
        cdef Py_ssize_t i, j
        cdef double precice_dt

        # Mesh setup
        mesh = None
        if config.get_mesh_type() == "CERFONS":
            mesh = Mesh(config, "/u/idesai/fusion-core-coupling/cerfons_geom_data.nc")
        elif config.get_mesh_type() == "CIRCULAR":
            mesh = Mesh(config, "/u/idesai/fusion-core-coupling/circular_geom_data.nc")

        cdef double drho = mesh.get_drho()
        cdef double dtheta = mesh.get_dtheta()
        nrho, ntheta = mesh.get_nrho(), mesh.get_ntheta()
        cdef double [::1] rho = mesh.get_rho_vals()
        cdef double [::1] theta = mesh.get_theta_vals()
        cdef double [:, ::1] xpol = mesh.get_x_vals()
        cdef double [:, ::1] ypol = mesh.get_y_vals()
        cdef double [:, ::1] jac = mesh.get_jacobian()
        cdef double [:, ::1] g_rr = mesh.get_g_rho_rho()
        cdef double [:, ::1] g_rt = mesh.get_g_rho_theta()
        cdef double [:, ::1]  g_tt = mesh.get_g_theta_theta()

        # Field variable array
        u_np = np.zeros((ntheta, nrho), dtype=np.double)
        cdef double [:, ::1] u = u_np
        du_np = np.zeros((ntheta, nrho), dtype=np.double)
        cdef double [:, ::1] du = du_np

        # Analytical solution setup
        ansol_bessel = Ansol(config)

        # Setting initial state of the field using analytical solution formulation
        for i in range(ntheta):
            for j in range(nrho):
                u[i, j] = ansol_bessel.ansol(rho[j], theta[i], 0)

        # Initialize boundary conditions at inner and outer edge of the torus
        bndvals_wall = np.zeros((mesh.get_n_points_wall(), 2))
        boundary = Boundary(config, mesh, bndvals_wall, u)

        # Reset boundary conditions according to analytical solution
        boundary.set_bnd_vals_ansol(ansol_bessel, u, 0)

        if coupling_on:
            # Setup read coupling mesh
            vertices = []
            for i in range(ntheta):
                vertices.append([xpol[i, nrho-1], ypol[i, nrho-1]])

            read_vertex_ids = self._interface.set_mesh_vertices(self._interface.get_mesh_id(
                self._config.get_read_mesh_name()), vertices)

            # Set up read mesh in preCICE
            read_mesh_id = self._interface.get_mesh_id(self._config.get_read_mesh_name())
            read_data_id = self._interface.get_data_id(self._config.get_read_data_name(), read_mesh_id)

            # Setup write coupling mesh
            vertices = []
            for i in range(ntheta):
                vertices.append([xpol[i, nrho-3], ypol[i, nrho-3]])

            write_vertex_ids = self._interface.set_mesh_vertices(self._interface.get_mesh_id(
                self._config.get_write_mesh_name()), vertices)

            # Set up write mesh in preCICE
            write_mesh_id = self._interface.get_mesh_id(self._config.get_write_mesh_name())
            write_data_id = self._interface.get_data_id(self._config.get_write_data_name(), write_mesh_id)

        # Get parameters from config and mesh modules
        diffc = config.get_diffusion_coeff()
        self.logger.info('Diffusion coefficient = %f', diffc)
        cdef double dt = config.get_dt()
        self.logger.info('dt = %f', dt)
        t_total, t_out = config.get_total_time(), config.get_t_output()

        # Check the CFL Condition for Diffusion Equation
        cfl_rho = dt * diffc / (drho * drho)
        self.logger.info('CFL Coefficient with radial param = %f. Must be less than 0.5', cfl_rho)
        cfl_theta = dt * diffc / (np.mean(rho) * np.mean(rho) * dtheta * dtheta)
        self.logger.info('CFL Coefficient with theta param = %f. Must be less than 0.5', cfl_theta)
        assert (cfl_rho < 0.5)
        assert (cfl_theta < 0.5)

        if coupling_on:
            # Initialize preCICE interface
            precice_dt = self._interface.initialize()
            dt = min(precice_dt, dt)

        cdef int n_t = int(t_total/dt)
        cdef int n_out = int(t_out/dt)
        print("n_t = {}, n_out = {}".format(n_t, n_out))

        # Write initial state
        write_csv(u, mesh, 0)
        write_vtk(u, mesh, 0)

        # Time loop
        cdef double u_sum
        # Time loop
        cdef int n = 0
        cdef double t = 0.0
        while self._interface.is_coupling_ongoing():
            if coupling_on:
                # Read data from preCICE and set fluxes
                flux_vals = self._interface.read_block_scalar_data(read_data_id, read_vertex_ids)
                boundary.set_bnd_vals_so(u, flux_vals)

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

            if coupling_on:
                # Write data to coupling interface preCICE
                node_vals = boundary.get_bnd_vals(u)
                self._interface.write_block_scalar_data(write_data_id, write_vertex_ids, node_vals)

                # Advance coupling via preCICE
                precice_dt = self._interface.advance(dt)

            # Set analytical boundary conditions in each iteration
            boundary.set_bnd_vals_ansol(ansol_bessel, u, (n+1)*dt)

            # update solution
            n += 1
            t += dt

            if n%n_out == 0 or n == n_t-1:
                write_csv(u, mesh, n+1)
                write_vtk(u, mesh, n+1)
                self.logger.info('VTK file output written at t = %f', n*dt)
                u_sum = 0
                for i in range(ntheta):
                    for j in range(nrho):
                        u_sum += u[i, j]

                self.logger.info("Elapsed time = {}  || Field sum = {}".format(n*dt, u_sum/(nrho*ntheta)))
                self.logger.info("Elapsed CPU time = {}".format(process_time()))

                ansol_bessel.compare_ansoln(mesh, u, n*dt, self.logger)

        self.logger.info("Total CPU time = {}".format(process_time()))
        # End
