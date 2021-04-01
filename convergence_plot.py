"""
Convergence plot for error of numerical simulation with analytical solution derived from Bessel functions
"""
import matplotlib.pyplot as plt
import numpy as np

x_data = [25, 50, 100, 200]
dr = [6.12E-03, 3.06E-03, 1.53E-03, 1.02E-03]
dtheta = [0.062832, 0.031416, 0.015708, 0.010472]
dt = [1.00E-03, 2.50E-04, 6.25E-05, 2.78E-05]

# Dirichlet BC results
coarse_error_dbc = 0.000363
# Manually enter errors acquired from runs
l2_error_dbc = [coarse_error_dbc, 0.0000854, 0.0000207, 0.0000051]
O1_error_dbc = [coarse_error_dbc, coarse_error_dbc/2, coarse_error_dbc/4, coarse_error_dbc/8]
O2_error_dbc = [coarse_error_dbc, coarse_error_dbc/4, coarse_error_dbc/16, coarse_error_dbc/64]

# Neumann - Dirichlet BC results
coarse_error_nbc = 0.0027
# Manually enter errors acquired from runs
l2_error_nbc = [coarse_error_nbc, 0.0006, 0.000143, 0.0000348]
O1_error_nbc = [coarse_error_nbc, coarse_error_nbc/2, coarse_error_nbc/4, coarse_error_nbc/8]
O2_error_nbc = [coarse_error_nbc, coarse_error_nbc/4, coarse_error_nbc/16, coarse_error_nbc/64]

plt.title('Dirichlet BCs on inner edge and Neumann BCs and outer edge')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('mesh size')
plt.ylabel('l2 error')

plt.plot(x_data, l2_error_nbc, 'bs', label="l2 error")
plt.plot(x_data, O1_error_nbc, 'r--', label="O(n^1) error", linewidth=1)
plt.plot(x_data, O2_error_nbc, 'g--', label="O(n^2) error", linewidth=1)
plt.legend(loc="upper right")
plt.show()
