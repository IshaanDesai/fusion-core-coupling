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

    for i in range(n_x):
        for j in range(n_x):
            x_out[i, j, 0] = coords[i, j, 0]
            y_out[i, j, 0] = coords[i, j, 1]
            z_out[i, j, 0] = 0
            u_out[i, j, 0] = u[i, j]

    gridToVTK("./output/"+filename+"_"+str(n), x_out, y_out, z_out, pointData={"value": u_out})

# General
filename = "diff-cart"

# Geometric quantities
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
dx = 1.0e-1
n_x = int((x_max - x_min) / dx) + 1
n_y = int((y_max - y_min) / dx) + 1

# Other physical quantities
diff_coeff = 0.01
dt = 1.0e-3
end_t = 1.0
t = 0
n = 0

# Generate a coordinate grid 
coords = np.zeros((n_x, n_y, 2), dtype=np.double)
for i in range(n_x):
    for j in range(n_y):
        coords[i, j, 0] = x_min + i*dx # uniform grid
        coords[i, j, 1] = y_min + j*dx # uniform grid

# Field variable having values for which the diffusion problem is solved
u = np.zeros((n_x, n_y), dtype=np.double)
du = np.zeros((n_x, n_y), dtype=np.double)

# Initial condition: Fluctuation in center of square plate
for i in range(int(n_x/2) - 5, int(n_x/2) + 5):
    for j in range(int(n_y/2) - 5, int(n_y/2) + 5):
        u[i, j] = 1.0

# Write intial condition
write_vtk(coords, u, n)

# Solve diffusion equation
while t < end_t:
    for i in range(1, n_x - 1):
        for j in range(1, n_y - 1):
            du[i, j] = (dt*diff_coeff / dx**2)*(u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - 4*u[i, j])

    # Update the values for next time step
    for i in range(1, n_x - 1):
        for j in range(1, n_y - 1):
            u[i, j] += du[i, j]

    # Update time
    n += 1
    t += dt

    print("t = {}".format(t))

    # output
    if n%100 == 0:
        write_vtk(coords, u, n)
