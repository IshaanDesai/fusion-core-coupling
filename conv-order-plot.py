"""
Convergence plot for error of numerical simulation with analytical solution derived from Bessel functions
"""
import matplotlib.pyplot as plt
import numpy as np

# Number of grid points in radial / x- direction
x_data = [25, 50, 100, 200]

"""
Parameters for Polar Code:
alpha = 1
radial mode number = 2
ums (for poloidal mode number = 2) = 8.4172
t_end = 0.015
r_min = 0.2
r_max = 1.0
rho_points = [25, 50, 100, 200]
theta_points = [100, 200, 400, 800]
dt           = [1.6e-5, 4.0e-6, 1.0e-6, 2.5e-7]
"""
# CORE: Polar Code (Dirichlet and Neumann BCs) results 
l2_errc_core = 3.0e-4
linf_errc_core = 0
# Manually enter errors acquired from runs
l2_core = [l2_errc_core, 8.54e-5, 2.07e-5, 5.1e-6]
linf_core = []
a, b = 1, 1
l2_o1_core, l2_o2_core = [], []
linf_o1_core, linf_o2_core = [], []
for i in range(4):
    l2_o1_core.append(l2_errc_core / a)
    l2_o2_core.append(l2_errc_core / b)
    linf_o1_core.append(linf_errc_core * a)
    linf_o2_core.append(linf_errc_core * b)
    a *= 2
    b *= 4

"""
Parameters for PARALLAX:
alpha = 1
radial mode number = 2
ums (for poloidal mode number = 2) = 8.4172
t_end = 0.015
r_min = 0.0
r_max = 1.0
grid_spacing_f = [3.2e-2, 1.6e-2, 8.0e-3, 4.0e-3]
dt             = [1.6e-5, 4.0e-6, 1.0e-6, 2.5e-7]
"""
# EDGE: PARALLAX (only Dirichlet BCs) results
l2_errc_edge = 3.1e-4
# Manually enter errors acquired from runs
linf_errc_edge = [5.7e-4, 1.4e-4, 3.6e-5, 4.8e-6]
l2_edge = [l2_errc_edge, 7.7e-5, 1.23 - 5, 3.0e-6]
linf_edge = []
a, b = 1, 1
l2_o1_edge, l2_o2_edge = [], []
linf_o1_edge, linf_o2_edge = [], []
for i in range(4):
    l2_o1_edge.append(l2_errc_edge / a)
    l2_o2_edge.append(l2_errc_edge / b)
    # linf_o1_edge.append(linf_errc_edge*a)
    # linf_o2_edge.append(linf_edge_core*b)
    a *= 2
    b *= 4

"""
Coupled case data for individual participants
"""
l2_err_core_c = [2.47e-2, 8.8e-3, 3.56e-3, 0.0]
linf_err_core_c = [5.91e-2, 2.37e-2, 9.68e-3]

l2_err_edge_c = [1.58e-3, 8.71e-3, 4.4e-3, 0.0]
linf_err_edge_c = [4.73e-3, 2.5e-2, 1.22e-2]

plt.title('Single physics Core and Edge codes convergence order')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('mesh size')
plt.ylabel('l2 error')

plt.plot(x_data, l2_core, 'rs', label="Single physics: core l2 error")
plt.plot(x_data, l2_o1_core, 'r-', label="Single physics: core O(n^1) error", linewidth=1)
plt.plot(x_data, l2_o2_core, 'r--', label="Single physics: core O(n^2) error", linewidth=1)
plt.plot(x_data, l2_edge, 'bs', label="Single physics: edge l2 error")
plt.plot(x_data, l2_o1_edge, 'b-', label="Single physics: edge O(n^1) error", linewidth=1)
plt.plot(x_data, l2_o1_edge, 'b--', label="Single physics: edge O(n^2) error", linewidth=1)
plt.plot(x_data, l2_err_core_c, 'g*', label="Coupled case: core l2 error")
plt.plot(x_data, l2_err_edge_c, 'g^', label="Coupled case: edge l2 error")
plt.legend(loc="upper right")
plt.show()
