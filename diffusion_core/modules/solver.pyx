"""
Solvers with specific space and time discretization stencils
"""
import numpy as np
cimport numpy as np
cimport cython
import math

cdef class Diffusion2D:
    def __init__(self, mesh):
        self.drho = mesh.get_drho()
        self.dtheta = mesh.get_dtheta()
        self.nrho = mesh.get_nrho()
        self.ntheta = mesh.get_ntheta()

        self.rho = mesh.get_rho_vals()
        self.theta = mesh.get_theta_vals()

        self.jac = mesh.get_jacobian()
        self.g_rr = mesh.get_g_rho_rho()
        self.g_rt = mesh.get_g_rho_theta()
        self.g_tt = mesh.get_g_theta_theta()

    def solve(self, dt, u):
        cdef double [:, ::1] du = np.zeros((self.ntheta, self.nrho), dtype=np.double)
        cdef double [:, ::1] u_np1 = np.zeros((self.ntheta, self.nrho), dtype=np.double)

        # Assign values to ghost cells for periodicity in theta direction
        ip = [self.ntheta - 2, self.ntheta - 1, 0, 1]
        for i in range(1, 3):
            for j in range(1, self.nrho - 1):
                # Pre-computing indices for speed-up
                ii = ip[i]
                i_p = ip[i+1]
                i_m = ip[i-1]
                j_p = j+1
                j_m = j-1

                # Staggered grid for rho-rho diagonal term
                du[ii, j] = ((self.jac[ii, j_p] + self.jac[ii, j])*(self.g_rr[ii, j_p] + self.g_rr[ii, j])*(u[ii, j_p] - u[ii, j]) -
                    (self.jac[ii, j] + self.jac[ii, j_m])*(self.g_rr[ii, j] + self.g_rr[ii, j_m])*(u[ii, j] - u[ii, j_m])) / (4*self.drho*self.drho)

                # Staggered grid for theta-theta diagonal term
                du[ii, j] += ((self.jac[i_p, j] + self.jac[ii, j])*(self.g_tt[i_p, j] + self.g_tt[ii, j])*(u[i_p, j] - u[ii, j]) -
                    (self.jac[ii, j] + self.jac[i_m, j])*(self.g_tt[ii, j] + self.g_tt[i_m, j])*(u[ii, j] - u[i_m, j])) / (4*self.dtheta*self.dtheta)

                # Off-diagonal term rho-theta
                du[ii, j] += (self.jac[ii, j_p]*self.g_rt[ii, j_p]*(u[i_p, j_p] - u[i_m, j_p]) -
                    self.jac[ii, j_m]*self.g_rt[ii, j_m]*(u[i_p, j_m] - u[i_m, j_m])) / (4*self.drho*self.dtheta)

                # Off-diagonal term theta-rho
                du[ii, j] += (self.jac[i_p, j]*self.g_rt[i_p, j]*(u[i_p, j_p] - u[i_p, j_m]) -
                    self.jac[i_m, j]*self.g_rt[i_m, j]*(u[i_m, j_p] - u[i_m, j_m])) / (4*self.dtheta*self.drho)

        # Iterate over all grid points
        for i in range(1, self.ntheta - 1):
            for j in range(1, self.nrho - 1):
                # Pre-computing indices for speed-up
                i_p = i+1
                i_m = i-1
                j_p = j+1
                j_m = j-1

                # Staggered grid for rho-rho diagonal term
                du[i, j] = ((self.jac[i, j_p] + self.jac[i, j])*(self.g_rr[i, j_p] + self.g_rr[i, j])*(u[i, j_p] - u[i, j]) -
                    (self.jac[i, j] + self.jac[i, j_m])*(self.g_rr[i, j] + self.g_rr[i, j_m])*(u[i, j] - u[i, j_m])) / (4*self.drho*self.drho)

                # Staggered grid for theta-theta diagonal term
                du[i, j] += ((self.jac[i_p, j] + self.jac[i, j])*(self.g_tt[i_p, j] + self.g_tt[i, j])*(u[i_p, j] - u[i, j]) -
                    (self.jac[i, j] + self.jac[i_m, j])*(self.g_tt[i, j] + self.g_tt[i_m, j])*(u[i, j] - u[i_m, j])) / (4*self.dtheta*self.dtheta)

                # Off-diagonal term rho-theta
                du[i, j] += (self.jac[i, j_p]*self.g_rt[i, j_p]*(u[i_p, j_p] - u[i_m, j_p]) -
                    self.jac[i, j_m]*self.g_rt[i, j_m]*(u[i_p, j_m] - u[i_m, j_m])) / (4*self.drho*self.dtheta)

                # Off-diagonal term theta-rho
                du[i, j] += (self.jac[i_p, j]*self.g_rt[i_p, j]*(u[i_p, j_p] - u[i_p, j_m]) -
                    self.jac[i_m, j]*self.g_rt[i_m, j]*(u[i_m, j_p] - u[i_m, j_m])) / (4*self.dtheta*self.drho)

        # Update scheme
        for i in range(self.ntheta):
            for j in range(self.nrho):
                u_np1[i, j] += dt*du[i, j] / self.jac[i, j]

        return u_np1