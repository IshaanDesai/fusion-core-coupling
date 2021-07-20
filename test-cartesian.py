"""
Diffusion equation solved in 2D on a circular domain with a Cartesian grid
"""
import numpy as np
import math
import precice
from pyevtk.hl import gridToVTK


def write_vtk(coords, u, n):
    print("Writing VTK output at n = {}".format(n))
    # Other variables for output
    x_out = np.zeros((n_x, n_y, 1), dtype=np.double)
    y_out = np.zeros((n_x, n_y, 1), dtype=np.double)
    z_out = np.zeros((n_x, n_y, 1), dtype=np.double)
    u_out = np.zeros((n_x, n_y, 1), dtype=np.double)

    counter = 0
    for i in range(n_x):
        for j in range(n_x):
            x_out[i, j, 0] = coords[counter][0]
            y_out[i, j, 0] = coords[counter][1]
            z_out[i, j, 0] = 0
            u_out[i, j, 0] = u[i, j]
            counter += 1

    gridToVTK("./output/" + filename + "_" + str(n), x_out, y_out, z_out, pointData={"value": u_out})


# General
filename = "diff-cart"
coupling_on = True  # Currently being used for a coupled case. Change to "False" for single physics run

# Geometric quantities
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
dx = 1.0e-1
n_x = int((x_max - x_min) / dx) + 1
n_y = int((y_max - y_min) / dx) + 1

# Other physical quantities
diff_coeff = 1.0
dt = 1.0e-4
end_t = 0.5
t = 0
n = 0

# Generate a grid
coords = []
for i in range(n_x):
    for j in range(n_y):
        coords.append([x_min + i * dx, y_min + j * dx])  # uniform grid

# Field variable having values for which the diffusion problem is solved
u = np.zeros((n_x, n_y), dtype=np.double)
du = np.zeros((n_x, n_y), dtype=np.double)

if coupling_on:
    # Define coupling interface
    interface = precice.Interface("Test-Cartesian", "test-precice-config.xml", 0, 1)

    # Setup coupling mesh
    mesh_id = interface.get_mesh_id("test-cart-mesh")
    vertex_ids = interface.set_mesh_vertices(interface.get_mesh_id("test-cart-mesh"), coords)
    read_data_id = interface.get_data_id("value", mesh_id)

    # Initialize preCICE interface
    precice_dt = interface.initialize()
    dt = min(precice_dt, dt)

# Initial condition: Fluctuation in center of square plate
# for i in range(int(n_x / 2) - 5, int(n_x / 2) + 5):
#     for j in range(int(n_y / 2) - 5, int(n_y / 2) + 5):
#         u[i, j] = 1.0

# Write intial condition
write_vtk(coords, u, n)

# Solve diffusion equation
while t < end_t:

    if coupling_on:
        # Read data from preCICE
        u_read = interface.read_block_scalar_data(read_data_id, vertex_ids)
        # Apply the read values to the field
        counter = 0
        for i in range(n_x):
            for j in range(n_y):
                u[i, j] = u_read[counter]
                counter += 1

    for i in range(1, n_x - 1):
        for j in range(1, n_y - 1):
            du[i, j] = (dt * diff_coeff / dx**2) * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j])

    # Update the values for next time step
    for i in range(1, n_x - 1):
        for j in range(1, n_y - 1):
            u[i, j] += du[i, j]

    if coupling_on:
        # Advance coupling via preCICE
        precice_dt = interface.advance(dt)

    # Update time
    n += 1
    t += dt

    print("t = {}".format(t))

    # output
    if n == int(0.5 / 1.0e-4):
        write_vtk(coords, u, n)
