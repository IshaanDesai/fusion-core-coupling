"""
Code to simulate diffusion in a polar coordinate system replicating a gyrokinetics fusion code (reactor core physics)
"""

import numpy as np
cimport numpy as np
cimport cython
from diffusion_core.modules.output import write_vtk, write_csv, write_custom_csv
from diffusion_core.modules.config import Config
from diffusion_core.modules.mesh import Mesh
from diffusion_core.modules.boundary import Boundary
import math
from time import process_time
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
        start_time = process_time()

        self.logger.info('Solving Diffusion case')
        # Read initial conditions from a JSON config file
        problem_config_file = "diffusion-coupling-config.json"
        config = Config(problem_config_file)

        # Iterators
        cdef Py_ssize_t i, j

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
        # Field explicit update array
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

        # Set boundary conditions (Dirichlet)
        for i in range(ntheta):
            for j in range(nrho):
                if j == 0 or j == nrho-1:
                    u[i, j] = 0.0

        # Set boundary conditions
        flux_vals = np.full((ntheta), 0.0)
        boundary = Boundary(config, mesh, flux_vals, u)

        # Setup read coupling mesh
        vertices = []
        for i in range(ntheta):
            vertices.append([xpol[i, nrho-3], ypol[i, nrho-3]])

        self._read_vertex_ids = self._interface.set_mesh_vertices(self._interface.get_mesh_id(
            self._config.get_read_mesh_name()), vertices)

        # Set up read mesh in preCICE
        self._read_mesh_id = self._interface.get_mesh_id(self._config.get_read_mesh_name())
        self._read_data_id = self._interface.get_data_id(self._config.get_read_data_name(), self._read_mesh_id)

        # Clear vertices list for reuse
        vertices.clear()

        # Setup write coupling mesh
        vertices = []
        for i in range(ntheta):
            vertices.append([xpol[i, nrho-1], ypol[i, nrho-1]])

        self._write_vertex_ids = self._interface.set_mesh_vertices(self._interface.get_mesh_id(
            self._config.get_write_mesh_name()), vertices)

        # Set up write mesh in preCICE
        self._write_mesh_id = self._interface.get_mesh_id(self._config.get_write_mesh_name())
        self._write_data_id = self._interface.get_data_id(self._config.get_write_data_name(), self._write_mesh_id)

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

        # Initialize preCICE interface
        cdef double precice_dt = self._interface.initialize()

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
            # Read data from preCICE and set fluxes
            flux_vals = self._interface.read_block_vector_data(self._read_data_id, self._read_vertex_ids)
            boundary.set_bnd_vals(u, flux_vals)

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

            # Write data to coupling interface preCICE
            node_vals = boundary.get_bnd_vals(u)
            self._interface.write_block_scalar_data(self._write_data_id, self._write_vertex_ids, node_vals)

            # Advance coupling via preCICE
            precice_dt = self._interface.advance(dt)

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
                self.logger.info("Elapsed CPU time = {}".format(process_time() - start_time))

        self._interface.finalize()
        self.logger.info("Total CPU time = {}".format(process_time() - start_time))
