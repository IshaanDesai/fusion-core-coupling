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
        cdef double [:, ::1] u = np.zeros((ntheta, nrho), dtype=np.double)
        cdef double [:, ::1] du = np.zeros((ntheta, nrho), dtype=np.double)
        cdef double [:, ::1] u_err = np.zeros((ntheta, nrho), dtype=np.double)

        # Analytical solution setup
        ansol_bessel = Ansol(config, mesh)

        # Setting initial state of the field using analytical solution formulation
        for i in range(ntheta):
            for j in range(nrho):
                u[i, j] = ansol_bessel.ansol(rho[j], theta[i], 0)

        # Initialize boundary conditions at inner and outer edge of the torus
        boundary = Boundary(mesh)

        # Reset boundary conditions according to analytical solution
        boundary.set_bnd_vals_ansol(u, ansol_bessel, 0)

        if coupling_on:
            # Define coupling interface
            interface = precice.Interface(config.get_participant_name(), config.get_config_file_name(), 0, 1)
            
            # Setup write coupling mesh (mutliple layers of polar mesh from interior of domain)
            vertices = []
            for j in range(nrho):
                for i in range(ntheta):
                    vertices.append([xpol[i, j], ypol[i, j]])

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
        assert (cfl_rho < 0.5)
        assert (cfl_theta < 0.5)

        if coupling_on:
            # Initialize preCICE interface
            precice_dt = interface.initialize()
            dt = min(precice_dt, dt)

        cdef int n_t = int(t_total/dt)
        cdef int n_out = int(t_out/dt)
        self.logger.info("n_t = {}, n_out = {}".format(n_t, n_out))

        # Write initial state
        write_vtk("fusion-core", u, mesh, 0)

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
                # Update time step
                dt = min(precice_dt, dt)

            # Assign values to ghost cells for periodicity in theta direction
            ip = [ntheta-2, ntheta-1, 0, 1]
            for i in range(1, 3):
                for j in range(1, nrho-1):
                    # Pre-computing indices for speed-up
                    ii = ip[i]
                    i_p = ip[i+1]
                    i_m = ip[i-1]
                    j_p = j+1
                    j_m = j-1

                    # Staggered grid for rho-rho diagonal term
                    du[ii, j] = ((jac[ii, j_p] + jac[ii, j])*(g_rr[ii, j_p] + g_rr[ii, j])*(u[ii, j_p] - u[ii, j]) -
                        (jac[ii, j] + jac[ii, j_m])*(g_rr[ii, j] + g_rr[ii, j_m])*(u[ii, j] - u[ii, j_m])) / (4*drho*drho)

                    # Staggered grid for theta-theta diagonal term
                    du[ii, j] += ((jac[i_p, j] + jac[ii, j])*(g_tt[i_p, j] + g_tt[ii, j])*(u[i_p, j] - u[ii, j]) -
                        (jac[ii, j] + jac[i_m, j])*(g_tt[ii, j] + g_tt[i_m, j])*(u[ii, j] - u[i_m, j])) / (4*dtheta*dtheta)

                    # Off-diagonal term rho-theta
                    du[ii, j] += (jac[ii, j_p]*g_rt[ii, j_p]*(u[i_p, j_p] - u[i_m, j_p]) -
                        jac[ii, j_m]*g_rt[ii, j_m]*(u[i_p, j_m] - u[i_m, j_m])) / (4*drho*dtheta)

                    # Off-diagonal term theta-rho
                    du[ii, j] += (jac[i_p, j]*g_rt[i_p, j]*(u[i_p, j_p] - u[i_p, j_m]) -
                        jac[i_m, j]*g_rt[i_m, j]*(u[i_m, j_p] - u[i_m, j_m])) / (4*dtheta*drho)

            # Iterate over all grid points
            for i in range(1, ntheta-1):
                for j in range(1, nrho-1):
                    # Pre-computing indices for speed-up
                    i_p = i+1
                    i_m = i-1
                    j_p = j+1
                    j_m = j-1

                    # Staggered grid for rho-rho diagonal term
                    du[i, j] = ((jac[i, j_p] + jac[i, j])*(g_rr[i, j_p] + g_rr[i, j])*(u[i, j_p] - u[i, j]) -
                        (jac[i, j] + jac[i, j_m])*(g_rr[i, j] + g_rr[i, j_m])*(u[i, j] - u[i, j_m])) / (4*drho*drho)

                    # Staggered grid for theta-theta diagonal term
                    du[i, j] += ((jac[i_p, j] + jac[i, j])*(g_tt[i_p, j] + g_tt[i, j])*(u[i_p, j] - u[i, j]) -
                        (jac[i, j] + jac[i_m, j])*(g_tt[i, j] + g_tt[i_m, j])*(u[i, j] - u[i_m, j])) / (4*dtheta*dtheta)

                    # Off-diagonal term rho-theta
                    du[i, j] += (jac[i, j_p]*g_rt[i, j_p]*(u[i_p, j_p] - u[i_m, j_p]) -
                        jac[i, j_m]*g_rt[i, j_m]*(u[i_p, j_m] - u[i_m, j_m])) / (4*drho*dtheta)

                    # Off-diagonal term theta-rho
                    du[i, j] += (jac[i_p, j]*g_rt[i_p, j]*(u[i_p, j_p] - u[i_p, j_m]) -
                        jac[i_m, j]*g_rt[i_m, j]*(u[i_m, j_p] - u[i_m, j_m])) / (4*dtheta*drho)

            # Update scheme
            for i in range(ntheta):
                for j in range(nrho):
                    u[i, j] += dt*du[i, j] / jac[i, j]

            # update solution
            n += 1
            t += dt

            if coupling_on:
                # Write data to coupling interface preCICE
                node_vals = boundary.get_bnd_vals(u)
                interface.write_block_scalar_data(write_data_id, write_vertex_ids, node_vals)

                # Advance coupling via preCICE
                precice_dt = interface.advance(dt)
            else:
                # Set analytical boundary conditions in each iteration
                boundary.set_bnd_vals_ansol(u, ansol_bessel, t)

            if n%n_out == 0 or n == n_t:
                # write_csv("fusion-core", u, mesh, n)
                write_vtk("fusion-core", u, mesh, n)
                self.logger.info('VTK file output written at t = %f', t)
                u_sum = 0
                for i in range(ntheta):
                    for j in range(nrho):
                        u_sum += u[i, j]
                        u_err[i, j] = abs(u[i, j] - ansol_bessel.ansol(rho[j], theta[i], t))

                self.logger.info("Elapsed time = {}  || Field sum = {}".format(n*dt, u_sum/(nrho*ntheta)))
                self.logger.info("Elapsed CPU time = {}".format(process_time()))

                ansol_bessel.compare_ansoln(u, n*dt, self.logger)

            # Simulation time is done
            if n >= n_t:
                is_coupling_ongoing = False
            
            if coupling_on:
                is_coupling_ongoing = interface.is_coupling_ongoing()

        if coupling_on:
            interface.finalize()
            
        self.logger.info("Total CPU time = {}".format(process_time()))
        # End
