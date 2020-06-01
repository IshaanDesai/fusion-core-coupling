"""
Code to simulate diffusion in a polar coordinate system replicating a gyrokinetics fusion code (reactor core physics)
"""

import numpy as np
from mesh_2d import Mesh

# Read initial conditions from a JSON config file
problem_config_file = "diffusion-coupling-config.json"

# Mesh setup
mesh = Mesh(problem_config_file)

# Initial condition setup
u = np.zeros(mesh.get_n_points_mesh())

# Time loop



# End
