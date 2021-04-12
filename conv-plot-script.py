"""
Convergence plot for error of numerical simulation with analytical solution derived from Bessel functions
"""
import matplotlib.pyplot as plt
import numpy as np

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
dt           = [1.6e-5, 4.0e-6, 1.0e-6, 2.5e-7, 6.25e-8]
"""
# CORE: Polar Code (Dirichlet and Neumann BCs) results 
coarse_err_core = 0.0003
# Manually enter errors acquired from runs
l2_err_core = [coarse_err_core, 0.0000854, 0.0000207, 0.0000051]
O1_err_core = [coarse_err_core, coarse_err_core/2, coarse_err_core/4, coarse_err_core/8]
O2_err_core = [coarse_err_core, coarse_err_core/4, coarse_err_core/16, coarse_err_core/64]

"""
Parameters for PARALLAX:
alpha = 1
radial mode number = 2
ums (for poloidal mode number = 2) = 8.4172
t_end = 0.015
r_min = 0.0
r_max = 1.0
grid_spacing_f = [3.2e-2, 1.6e-2, 8.0e-3, 4.0e-3, 2.0e-3]
dt             = [1.6e-5, 4.0e-6, 1.0e-6, 2.5e-7, 6.25e-8]
"""
# EDGE: PARALLAX (only Dirichlet BCs) results
coarse_err_edge = 0.00031
# Manually enter errors acquired from runs
linf_err_edge = [0.00057, 0.00014, 0.000036, 0.0000048]
l2_err_edge = [coarse_err_edge, 0.000077, 0.000012, 0.000003]
O1_err_edge = [coarse_err_edge, coarse_err_edge/2, coarse_err_edge/4, coarse_err_edge/8]
O2_err_edge = [coarse_err_edge, coarse_err_edge/4, coarse_err_edge/16, coarse_err_edge/64]

plt.title('Single physics Core and Edge codes convergence order')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('mesh size')
plt.ylabel('l2 error')

plt.plot(x_data, l2_err_core, 'rs', label="Polar Code l2 error")
plt.plot(x_data, O1_err_core, 'r-', label="Polar Code O(n^1) error", linewidth=1)
plt.plot(x_data, O2_err_core, 'r--', label="Polar Code O(n^2) error", linewidth=1)
plt.plot(x_data, l2_err_edge, 'bs', label="PARALLAX l2 error")
plt.plot(x_data, O1_err_edge, 'b-', label="PARALLAX O(n^1) error", linewidth=1)
plt.plot(x_data, O2_err_edge, 'b--', label="PARALLAX O(n^2) error", linewidth=1)
plt.legend(loc="upper right")
plt.show()
