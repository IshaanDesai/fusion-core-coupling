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


class BoundaryType(enum.Enum):
    """
    Defines boundary types: DIRICHLET = Dirichlet, NEUMANN_FO, NEUMANN_SO = Neumann first and second order.
    """
    DIRICHLET = 247
    NEUMANN_FO = 333
    NEUMANN_SO = 453


class Boundary:
    def __init__(self, config, mesh, data_neumann, field):
        self._nrho = mesh.get_nrho()
        self._ntheta = mesh.get_ntheta()
        self._drho = mesh.get_drho()
        self._dtheta = mesh.get_dtheta()
        self._g_rr = mesh.get_g_rho_rho()
        self._g_rt = mesh.get_g_rho_theta()
        self._g_tt = mesh.get_g_theta_theta()

        # Set boundary condition at initialization
        self.set_bnd_vals_so(field, data_neumann)

    def set_bnd_vals_fo(self, field, flux):
        j = self._nrho - 1
        for i in range(self._ntheta):
            # Dirichlet condition at inner boundary
            field[i, 0] = 0.0
            # Neumann condition at outer boundary (1st Order)
            field[i, j] = self._drho*flux[i] / math.sqrt(self._g_rr[i, j]) + field[i, j-1] + (self._g_rt[i, j]*self._drho)*(field[i-1, j-1] - field[i, j-1]) / (self._g_rr[i, j]*self._dtheta)

    def set_bnd_vals_so(self, field, flux):
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
