"""
This module implements boundary conditions on the boundary points of a given mesh.
Dirichlet and Neumann boundary conditions can be implemented.
For Neumann boundary conditions, the flux normal is in the inward directions (towards the center of polar grid).

NOTE: Neumann boundary conditions are implemented with the assumption that only Wall Boundary points will have
      Neumann conditions.
"""
import enum
import numpy as np
import math

class Boundary:
    def __init__(self, config, mesh):
        self._nrho = mesh.get_nrho()
        self._ntheta = mesh.get_ntheta()
        self._drho = mesh.get_drho()
        self._dtheta = mesh.get_dtheta()
        self._g_rr = mesh.get_g_rho_rho()
        self._g_rt = mesh.get_g_rho_theta()
        self._g_tt = mesh.get_g_theta_theta()
        self._rho = mesh.get_rho_vals()
        self._theta = mesh.get_theta_vals()

    def set_bnd_vals_so(self, field, flux):
        # Set the boundary values at the outer edge of the Core domain
        j = self._nrho - 1

        # Handle periodicity in theta direction due to symmetric stencil
        ip = [self._ntheta-2, self._ntheta-1, 0, 1]
        for i in range(1, 3):
            # Dirichlet condition at inner boundary
            field[ip[i], 0] = 0.0
            # Neumann condition at outer boundary (2nd order)
            field[ip[i], j] = 4*field[ip[i], j-1]/3 - field[ip[i], j-2]/3 - (self._drho*self._g_rt[ip[i], j])/(3*self._dtheta*self._g_rr[ip[i], j])*(2*field[ip[i-1], j-1] - 2*field[ip[i+1], j-1] + field[ip[i+1], j-2] - field[ip[i-1],j-2]) + \
                (2*self._drho)/(3*math.sqrt(self._g_rr[ip[i], j]))*(flux[ip[i]])

        for i in range(1, self._ntheta-1):
            # Dirichlet condition at inner boundary
            field[i, 0] = 0.0
            # Neumann condition at outer boundary (2nd order)
            field[i, j] = 4*field[i, j-1]/3 - field[i, j-2]/3 - (self._drho*self._g_rt[i, j])/(3*self._dtheta*self._g_rr[i, j])*(2*field[i-1, j-1] - 2*field[i+1, j-1] + field[i+1, j-2] - field[i-1,j-2]) + \
                (2*self._drho)/(3*math.sqrt(self._g_rr[i, j]))*(flux[i])

    def get_bnd_vals(self, field):
        bnd_data = []
        # Write data from the interior of the domain (2 mesh widths inside the physical boundary)
        j = self._nrho - 3
        # Gets Dirichlet values and returns them for coupling
        for i in range(self._ntheta):
            bnd_data.append(field[i, j])
        
        return np.array(bnd_data)

    def set_bnd_vals_ansol(self, ansol, field, t):
        """
        Assign Neumann boundary condition according to Bessel function at inner and outer boundary
        """
        # Set the boundary values at the outer edge of the Core domain
        j = self._nrho - 1

        # Handle periodicity in theta direction due to symmetric stencil
        ip = [self._ntheta-2, self._ntheta-1, 0, 1]
        for i in range(1, 3):
            # Dirichlet condition at inner boundary
            field[ip[i], 0] = ansol.ansol(self._rho[0], self._theta[i], t)
            # Neumann condition at outer boundary (2nd order)
            flux = ansol.ansol_gradient(self._rho[j], self._theta[i], t)
            field[ip[i], j] = 4*field[ip[i], j-1]/3 - field[ip[i], j-2]/3 - (self._drho*self._g_rt[ip[i], j])/(3*self._dtheta*self._g_rr[ip[i], j])*(2*field[ip[i-1], j-1] - 2*field[ip[i+1], j-1] + field[ip[i+1], j-2] - field[ip[i-1],j-2]) + \
                (2*self._drho)/(3*math.sqrt(self._g_rr[ip[i], j]))*flux

        for i in range(1, self._ntheta-1):
            # Dirichlet condition at inner boundary
            field[i, 0] = ansol.ansol(self._rho[0], self._theta[i], t)
            # Neumann condition at outer boundary (2nd order)
            flux = ansol.ansol_gradient(self._rho[j], self._theta[i], t)
            field[i, j] = 4*field[i, j-1]/3 - field[i, j-2]/3 - (self._drho*self._g_rt[i, j])/(3*self._dtheta*self._g_rr[i, j])*(2*field[i-1, j-1] - 2*field[i+1, j-1] + field[i+1, j-2] - field[i-1,j-2]) + \
                (2*self._drho)/(3*math.sqrt(self._g_rr[i, j]))*flux
