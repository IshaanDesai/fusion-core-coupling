"""
Code to simulate diffusion in a polar coordinate system replicating a gyrokinetics fusion code (reactor core physics).
Verification with analytical solution obtained using Bessel functions of the first kind
"""

import numpy as np
cimport numpy as np
cimport cython
from diffusion_core.modules.output import write_vtk, write_csv, write_custom_csv
from diffusion_core.modules.config import Config
from diffusion_core.modules.boundary cimport Boundary
from diffusion_core.modules.ansol cimport Ansol
from diffusion_core.modules.mesh import Mesh
from diffusion_core.modules.solver import Diffusion2D
import math
from time import process_time
import logging
import netCDF4 as nc
import precice
import argparse

class Diffusion:
    def __init__(self):
        self.logger = logging.getLogger('main.diffusion_core.Diffusion')

    def solve_diffusion(self):
        start_time = process_time()

        self.logger.info('Solving Diffusion case')

        parser = argparse.ArgumentParser(description="Solving diffusion equation on a polar coordinate system")
        parser.add_argument('filename', help='a string carrying the JSON config file name')

        args = parser.parse_args()

        # Read initial conditions from a JSON config file
        config = Config(args.filename)

        # Check if the case is coupling or not
        coupling_on = config.is_coupling_on()

        # Iterators
        cdef Py_ssize_t i, j
        cdef double precice_dt

        # Mesh setup
        mesh = Mesh(config)

        # Define solver
        diffusion_solver = Diffusion2D(mesh)

        cdef double drho = mesh.get_drho()
        cdef double dtheta = mesh.get_dtheta()
        nrho, ntheta = mesh.get_nrho(), mesh.get_ntheta()
        cdef double [:, ::1] xpol = mesh.get_x_vals()
        cdef double [:, ::1] ypol = mesh.get_y_vals()
        cdef double [::1] rho = mesh.get_rho_vals()
        cdef double [::1] theta = mesh.get_theta_vals()

        # Field variable array
        cdef double [:, ::1] u = np.zeros((ntheta, nrho), dtype=np.double)
        cdef double [:, ::1] du = np.zeros((ntheta, nrho), dtype=np.double)
        cdef double [:, ::1] u_err = np.zeros((ntheta, nrho), dtype=np.double)

        # Analytical solution setup
        ansol_bessel = Ansol(config, mesh)

        # Setting initial state of the field using analytical solution formulation
        for i in range(ntheta):
            for j in range(nrho):
                u[i, j] = ansol_bessel.ansol(rho[j], theta[i], 0)

        # Set rho values
        rho_write = mesh.get_rhomax() - 5 * drho
        rho_min = rho_write - 5 * drho
        rho_max = rho_write + 5 * drho
        # Constant overlap
        #rho_min = rho_write - 0.16
        #rho_max = rho_write + 0.16

        # Initialize boundary conditions at inner and outer edge of the torus
        boundary = Boundary(config, mesh, rho_min, rho_max, rho_write)

        # Reset boundary conditions according to analytical solution
        u = boundary.set_bnd_vals_so(u, ansol_bessel, 0)

        if coupling_on:
            # Define coupling interface
            interface = precice.Interface(config.get_participant_name(), config.get_config_file_name(), 0, 1)
            
            # Setup read coupling mesh
            vertices = []
            j = nrho - 1
            for i in range(ntheta):
                vertices.append([xpol[i, j], ypol[i, j]])

            self.logger.info('Read mesh has %d vertices', len(vertices))
            read_vertex_ids = interface.set_mesh_vertices(interface.get_mesh_id(config.get_read_mesh_name()), vertices)

            # Set up read mesh in preCICE
            read_mesh_id = interface.get_mesh_id(config.get_read_mesh_name())
            read_data_id = interface.get_data_id(config.get_read_data_name(), read_mesh_id)

            # Setup write coupling mesh
            vertices = []
            for j in range(nrho):
                for i in range(ntheta):
                    if rho[j] > rho_min and rho[j] < rho_max:
                        vertices.append([xpol[i, j], ypol[i, j]])

            self.logger.info('Write mesh has %d vertices', len(vertices))
            write_vertex_ids = interface.set_mesh_vertices(interface.get_mesh_id(config.get_write_mesh_name()), vertices)

            # Set up write mesh in preCICE
            write_mesh_id = interface.get_mesh_id(config.get_write_mesh_name())
            write_data_id = interface.get_data_id(config.get_write_data_name(), write_mesh_id)

        # Get parameters from config and mesh modules
        cdef double dt = config.get_dt()
        self.logger.info('dt = %f', dt)
        t_total, t_out = config.get_total_time(), config.get_t_output()

        # Check the CFL Condition for Diffusion Equation
        cfl_rho = dt / (drho * drho)
        self.logger.info('CFL Coefficient with radial param = %f. Must be less than 0.5', cfl_rho)
        cfl_theta = dt / (np.mean(rho) * np.mean(rho) * dtheta * dtheta)
        self.logger.info('CFL Coefficient with theta param = %f. Must be less than 0.5', cfl_theta)
        # assert (cfl_rho < 0.5)
        # assert (cfl_theta < 0.5)

        if coupling_on:
            # Initialize preCICE interface
            precice_dt = interface.initialize()
            dt = min(precice_dt, dt)

            if interface.is_action_required(precice.action_write_initial_data()):
                write_vals = boundary.get_analytical_bnd_vals(ansol_bessel, u, t)
                interface.write_block_scalar_data(write_data_id, write_vertex_ids, write_vals)
            
            interface.mark_action_fulfilled(precice.action_write_initial_data())

            interface.initialize_data()

        cdef int n_t = int(t_total/dt)
        cdef int n_out = int(t_out/dt)
        self.logger.info("n_t = {}, n_out = {}".format(n_t, n_out))

        # Write initial state
        # write_csv("fusion-core", u, mesh, 0)
        # write_vtk(self.logger, "fusion-core", u, mesh, 0)

        cdef double u_sum
        # Time loop
        cdef int n = 0
        cdef double t = 0.0

        if coupling_on:
            is_coupling_ongoing = interface.is_coupling_ongoing()
        else:
            is_coupling_ongoing = True

        while is_coupling_ongoing:
            if coupling_on:
                # Read data from preCICE and set fluxes (bi-directional coupling)
                flux_vals = interface.read_block_scalar_data(read_data_id, read_vertex_ids)
                # u = boundary.set_bnd_vals_so(u, ansol_bessel, t, flux_vals)
                boundary.compare_bnd_flux_vals(self.logger, u, ansol_bessel, t, flux_vals)

                # Update time step
                dt = min(precice_dt, dt)

            # Not solving problem in investigation of data mapping
            # u = diffusion_solver.solve(dt, u)            

            # update solution
            n += 1
            t += dt

            if coupling_on:
                # Write data to coupling interface preCICE
                # write_vals = boundary.get_bnd_vals(u)
                write_vals = boundary.get_analytical_bnd_vals(ansol_bessel, u, t)
                interface.write_block_scalar_data(write_data_id, write_vertex_ids, write_vals)

                # Advance coupling via preCICE
                precice_dt = interface.advance(dt)
            else:
                # Set analytical boundary conditions in each iteration
                u = boundary.set_bnd_vals_so(u, ansol_bessel, t)

            if n%n_out == 0 or n == n_t:
                # write_csv("fusion-core", u, mesh, n)
                # write_vtk(self.logger, "fusion-core", u, mesh, n)
                u_sum = 0
                for i in range(ntheta):
                    for j in range(nrho):
                        u_sum += u[i, j]
                        u_err[i, j] = abs(u[i, j] - ansol_bessel.ansol(rho[j], theta[i], t))

                # write_csv("error-inf", u_err, mesh, n)

                self.logger.info("Elapsed time = {}  || Field sum = {}".format(n*dt, u_sum/(nrho*ntheta)))
                self.logger.info("Elapsed CPU time = {}".format(process_time()))

                # ansol_bessel.compare_ansoln(u, n*dt, self.logger)
                # boundary.compare_bnd_flux_vals(u, ansol_bessel, t, flux_vals, self.logger)

            # Simulation time is done
            if n >= n_t:
                is_coupling_ongoing = False
            
            if coupling_on:
                is_coupling_ongoing = interface.is_coupling_ongoing()

        if coupling_on:
            interface.finalize()
            
        self.logger.info("Total CPU time = {}".format(process_time()))
        # End
