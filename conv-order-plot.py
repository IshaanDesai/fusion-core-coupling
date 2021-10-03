"""
Convergence plot for error of numerical simulation with analytical solution derived from Bessel functions
"""
import matplotlib.pyplot as plt

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
l2_core = [2.73e-3, 6.19e-4, 1.47e-4, 3.59e-5]
linf_core = [4.71e-3, 1.12e-3, 2.74e-4, 6.75e-5]

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
l2_edge = [3.1e-4, 7.69e-5, 1.91e-5, 4.77e-6]
linf_edge = [5.71e-4, 1.43e-4, 3.56e-5, 8.91e-6]

"""
Coupled case data for individual participants
r_interface = 0.7
Rest everything same as above individual data sets
"""
l2_core_c = []
linf_core_c = []

l2_edge_c = []
linf_edge_c = []

plt.title('Single physics Core and Edge codes convergence order')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('mesh size')
plt.ylabel('l2 error')

plt.plot(x_data, l2_core, 'rs', label="Single physics: core l2 error")
#plt.plot(x_data, l2_o1_core, 'r-', label="Single physics: core O(n^1) error", linewidth=1)
#plt.plot(x_data, l2_o2_core, 'r--', label="Single physics: core O(n^2) error", linewidth=1)
plt.plot(x_data, l2_edge, 'bs', label="Single physics: edge l2 error")
#plt.plot(x_data, l2_o1_edge, 'b-', label="Single physics: edge O(n^1) error", linewidth=1)
#plt.plot(x_data, l2_o1_edge, 'b--', label="Single physics: edge O(n^2) error", linewidth=1)
#plt.plot(x_data, l2_err_core_c, 'g*', label="Coupled case: core l2 error")
#plt.plot(x_data, l2_err_edge_c, 'g^', label="Coupled case: edge l2 error")
plt.legend(loc="upper right")
plt.show()
