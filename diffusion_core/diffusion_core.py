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
    if math.fabs(exponent) < 1E-10:
        gaussian = 0.0
    else:
        gaussian = math.exp(exponent)
    return gaussian


# Read initial conditions from a JSON config file
problem_config_file = "diffusion-coupling-config.json"
config = Config(problem_config_file)

# Mesh setup
mesh = Mesh(problem_config_file)
nx, ny = mesh.get_n_points_axiswise()
x_coords, y_coords = mesh.get_cart_coords()

print("Cartesian representation of polar mesh is 2D grid [{} x {}]".format(nx, ny))

# Definition of field variable
u = np.zeros((nx, ny))

# Setup Dirichlet boundary conditions at inner and outer edge of the torus
for i in range(nx):
    for j in range(ny):
        # Point type is a list and not a numpy array so it requires different index accessing
        if mesh.get_point_type(i, j) == MeshVertexType.GHOST:
            u[i, j] = 0

# Initialising Gaussian blob as initial condition of field
rhob = 2*math.pi / config.get_rhob()
wrhob = 2*math.pi / config.get_wrhob()
print("Center of Gaussian blob is at r = {}, rho = {}".format(config.get_rb(), rhob))
print("Width of Gaussian blob is wr = {}, wrho = {}".format(config.get_wrb(), wrhob))

for l in range(mesh.get_n_points_grid()):
    mesh_ind = mesh.grid_to_mesh_index(l)
    x = mesh.get_x(mesh_ind)
    y = mesh.get_y(mesh_ind)
    gaussx = gaussian_blob(config.get_rb(), config.get_wrb(), x)
    gaussy = gaussian_blob(rhob, wrhob, y)

    i, j = mesh.get_i_j_from_index(mesh_ind)
    u[i, j] = gaussx*gaussy

# Write data at initial condition
write_vtk(u, 0, x_coords, y_coords)

# Get parameters from config and mesh modules
diffc_perp = config.get_diffusion_coeff()
dt = config.get_dt()
n_t, n_out = config.get_n_timesteps(), config.get_n_output()
dr, drho = mesh.get_r_spacing(), mesh.get_rho_spacing()

# Time loop
for n in range(n_t):
    for l in range(mesh.get_n_points_grid()):
        mesh_ind = mesh.grid_to_mesh_index(l)
        l_i, l_j = mesh.get_i_j_from_index(mesh_ind)

        lrho_minus_i, lrho_minus_j = mesh.get_neighbor_index(mesh_ind, -1, 0)
        lrho_plus_i, lrho_plus_j = mesh.get_neighbor_index(mesh_ind, 1, 0)
        lr_minus_i, lr_minus_j = mesh.get_neighbor_index(mesh_ind, 0, -1)
        lr_plus_i, lr_plus_j = mesh.get_neighbor_index(mesh_ind, 0, 1)

        du_perp = (u[lr_minus_i, lr_minus_j] + u[lr_plus_i, lr_plus_j]) / (dr * dr)
        du_perp = du_perp + (u[lrho_minus_i, lrho_minus_j] + u[lrho_plus_i, lrho_plus_j]) / (drho * drho)
        du_perp = du_perp - 2 * u[l_i, l_j] * ((1 / (dr * dr)) + (1 / (drho * drho)))
        du_perp = du_perp*dt*diffc_perp

        u[l_i, l_j] += du_perp

    if n % n_out == 0:
        write_vtk(u, n, x_coords, y_coords)

    n += 1

# End
