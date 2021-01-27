"""
Read mesh from NetCFD4 file generated using the PARALLAX code
"""
import netCDF4 as nc
import numpy as np


class Mesh:
    def __init__(self, config, filename):
        ds = nc.Dataset(filename)
        self._drho, self._dtheta = ds.getncattr('drho'), ds.getncattr('dtheta')
        self._nrho, self._ntheta = ds.dimensions['nrho'].size, ds.dimensions['ntheta'].size
        assert self._nrho == config.get_rho_points()
        assert self._ntheta == config.get_theta_points()

        self._rho, self._theta = np.array(ds['rho'][:]), np.array(ds['theta'][:])
        self._x, self._y = np.array(ds['xpol'][:]), np.array(ds['ypol'][:])
        self._jac = np.array(ds['jacobian'][:])
        self._g_rr, self._g_rt = np.array(ds['g_rhorho'][:]), np.array(ds['g_rhotheta'][:])
        self._g_tt = np.array(ds['g_thetatheta'][:])

    def get_drho(self):
        return self._drho

    def get_dtheta(self):
        return self._dtheta

    def get_nrho(self):
        return self._nrho

    def get_ntheta(self):
        return self._ntheta

    def get_rho_vals(self):
        return self._rho

    def get_theta_vals(self):
        return self._theta

    def get_x_vals(self):
        return self._x

    def get_y_vals(self):
        return self._y

    def get_jacobian(self):
        return self._jac

    def get_g_rho_rho(self):
        return self._g_rr

    def get_g_rho_theta(self):
        return self._g_rt

    def get_g_theta_theta(self):
        return self._g_tt
