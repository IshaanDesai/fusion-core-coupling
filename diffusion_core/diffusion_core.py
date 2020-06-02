"""
Code to simulate diffusion in a polar coordinate system replicating a gyrokinetics fusion code (reactor core physics)
"""

import numpy as np
from mesh_2d import Mesh, MeshVertexType
from output import write_vtk

# Read initial conditions from a JSON config file
problem_config_file = "diffusion-coupling-config.json"

# Mesh setup
mesh = Mesh(problem_config_file)
nx, ny = mesh.get_n_points_axiswise()
x, y = mesh.get_cart_coords()
# Initial condition setup


u = np.zeros_like(x)
t = 0
write_vtk(u, t, x, y)

# Setup Dirichlet boundary conditions at inner and outer edge of the torus


# Time loop


# End
