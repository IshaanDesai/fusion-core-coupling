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
        self._rhomin = config.get_rhomin()
        self._rhomax = config.get_rhomax()
        self._drho = mesh.get_drho()
        self._dtheta = mesh.get_dtheta()
        self._g_rr = mesh.get_g_rho_rho()
        self._g_rt = mesh.get_g_rho_theta()
        self._g_tt = mesh.get_g_theta_theta()

        self._bnd_inds = []
        self._r, self._x, self._y, self._theta = [], [], [], []

        self.set_bnd_vals(field, data_neumann)

    def set_bnd_vals(self, field, data_neumann):
        counter = 0
        for i in range(self._nrho):
            for j in range(self._ntheta):
                if j == 0:
                    # Dirichlet condition at inner boundary
                    field[i, j] = 0.0
                elif j == self._nrho - 1:
                    flux = data_neumann[counter]
                    # Neumann condition at outer boundary (1st Order)
                    field[i, j] = self._drho*flux / math.sqrt(self._g_rr[i, j]) + field[i-1, j] + \
                                  (self._g_rt[i, j]*self._drho)*(field[i-1, j-1] - field[i-1, j]) / (self._g_rr[i, j]*self._dtheta)
                    counter += 1

    def get_bnd_vals(self, field):
        bnd_data = []
        # Gets Dirichlet values and returns them for coupling
        for i in range(self._nrho):
            for j in range(self._ntheta):
                if j == self._nrho - 1:
                    bnd_data.append(field[i, j])

        return np.array(bnd_data)

    def set_bnd_vals_mms(self, field, t):
        """
        gradient(f)_{r} = (2*pi/(rmax - rmin))*cos(2*pi*(r - rmin)/(rmax - rmin))*cos(t)*cos(theta)
        """
        counter = 0
        if self._bnd_type == BoundaryType.DIRICHLET:
            for inds in self._bnd_inds:
                # Zero value selected for Dirichlet boundary condition for MMS analysis
                field[inds[0], inds[1]] = 0.0
                counter += 1
        elif self._bnd_type == BoundaryType.NEUMANN_SO:
            for inds in self._bnd_inds:
                # Modify boundary value by evaluation of gradient of ansatz
                a = 2 * math.pi * (self._r[counter] - self._rmin) / (self._rmax - self._rmin)
                b = 2 * math.pi / (self._rmax - self._rmin)
                # NOTE: This implementation is only valid for Wall boundary cells (outer most cells)
                flux = b * math.cos(a) * math.cos(t) * math.cos(self._theta[counter])
                # Modify boundary value by second order evaluation of gradient
                field[inds[0], inds[1]] = (4/3)*field[inds[0]-1, inds[1]] - (1/3)*field[inds[0]-2, inds[1]] + (2/3)*self._dr*flux
                counter += 1
