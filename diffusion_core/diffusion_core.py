"""
Code to simulate diffusion in a polar coordinate system replicating a gyrokinetics fusion code (reactor core physics)
"""

import numpy as np
from mesh_2d import Mesh, MeshVertexType
from output import write_vtk
from config import Config
import math


def gaussian_blob(pos, wblob, coord):
    exponent = -1.0*(coord - pos)*(coord - pos) / (wblob*wblob)
    gaussian = math.exp(exponent)
    return gaussian


# Read initial conditions from a JSON config file
problem_config_file = "diffusion-coupling-config.json"
config = Config(problem_config_file)

# Mesh setup
mesh = Mesh(problem_config_file)
nx, ny = mesh.get_n_points_axiswise()
rmin, rmax = config.get_rmin(), config.get_rmax()

# Definition of field variable
u = np.zeros((nx, ny))

# Initialising Gaussian blob as initial condition of field
x_center, y_center = config.get_xb_yb()
x_width, y_width = config.get_wxb_wyb()
for l in range(mesh.get_n_points_grid()):
    mesh_ind = mesh.grid_to_mesh_index(l)
    x = mesh.get_x(mesh_ind)
    y = mesh.get_y(mesh_ind)
    gaussx = gaussian_blob(x_center, x_width, x)
    gaussy = gaussian_blob(y_center, y_width, y)

    i, j = mesh.get_i_j_from_index(mesh_ind)
    u[i, j] = gaussx*gaussy
    assert(u[i, j] >= 0), "gaussx = {}, gauss_y = {}, for ({},{})".format(gaussx, gaussx, i, j)

# Setup Dirichlet boundary conditions at inner and outer edge of the torus
for i in range(nx):
    for j in range(ny):
        # Point type is a list and not a numpy array so it requires different index accessing
        if mesh.get_point_type(i, j) == MeshVertexType.GHOST:
            u[i, j] = 0

# Write data at initial condition
write_vtk(u, mesh, 0)

# Get parameters from config and mesh modules
diffc_perp = config.get_diffusion_coeff()
print("Diffusion coefficient = {}".format(diffc_perp))
dt = config.get_dt()
print("dt = {}".format(dt))
n_t, n_out = config.get_n_timesteps(), config.get_n_output()
dr, dtheta = mesh.get_r_spacing(), mesh.get_theta_spacing()

# Check the CFL Condition for Diffusion Equation
cfl_r = dt*diffc_perp / (dr*dr)
print("CFL Coefficient with radial param = {}. Must be less than 0.5".format(cfl_r))
assert(cfl_r < 0.5)

cfl_theta = dt*diffc_perp / (rmin*rmin*dtheta*dtheta)
print("CFL Coefficient with theta param = {}. Must be less than 0.5".format(cfl_theta))
assert(cfl_theta < 0.5)

# Time loop
for n in range(n_t):
    for l in range(mesh.get_n_points_grid()):
        mesh_ind = mesh.grid_to_mesh_index(l)
        l_i, l_j = mesh.get_i_j_from_index(mesh_ind)
        r = mesh.get_r(mesh_ind)

        ltheta_minus_i, ltheta_minus_j = mesh.get_neighbor_i_j(mesh_ind, -1, 0)
        ltheta_plus_i, ltheta_plus_j = mesh.get_neighbor_i_j(mesh_ind, 1, 0)
        lr_minus_i, lr_minus_j = mesh.get_neighbor_i_j(mesh_ind, 0, -1)
        lr_plus_i, lr_plus_j = mesh.get_neighbor_i_j(mesh_ind, 0, 1)

        rminus_ind, rplus_ind = mesh.get_neighbor_index(mesh_ind, 0, -1), mesh.get_neighbor_index(mesh_ind, 0, 1)
        rminus, rplus = mesh.get_r(rminus_ind), mesh.get_r(rplus_ind)
        rplus_half, rminus_half = (rplus + r) / 2, (rminus + r) / 2

        # Staggered scheme to evaluate derivatives in radial direction
        du_perp = (rplus_half*(u[lr_plus_i, lr_plus_j] - u[l_i, l_j]) / dr - rminus_half*(u[lr_minus_i, lr_minus_j] - u[l_i, l_j]) / dr) / (r*dr)

        # Second order central difference components in theta direction
        du_perp += (u[ltheta_minus_i, ltheta_minus_j] + u[ltheta_plus_i, ltheta_plus_j] - 2*u[l_i, l_j]) / (r*r*dtheta*dtheta)

        du_perp = du_perp*dt*diffc_perp

        u[l_i, l_j] += du_perp

    if n % n_out == 0:
        write_vtk(u, mesh, n)

    n += 1

# End
