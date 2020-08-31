import matplotlib.pyplot as plt
import numpy as np

x_data = [50, 100, 200, 300]
dr = [6.12E-03, 3.06E-03, 1.53E-03, 1.02E-03]
dtheta = [0.062832, 0.031416, 0.015708, 0.010472]
dt = [1.00E-03, 2.50E-04, 6.25E-05, 2.78E-05]

# Dirichlet BC results
l2_error_dbc = [0.000418, 0.000097, 0.000023, 0.00001]
O1_error_dbc = [0.000418, 0.000209, 0.0001045, 0.00006966666667]
O2_error_dbc = [0.000418, 0.0001045, 0.000026125, 0.00001161111111]

# Neumann - Dirichlet BC results
l2_error_nbc = [0.014491, 0.003509, 0.000863, 0.000381]
O1_error_nbc = [0.014491, 0.0072455, 0.00362275, 0.002415166667]
O2_error_nbc = [0.014491, 0.00362275, 0.0009056875, 0.0004025277778]

plt.xscale('log')
plt.yscale('log')
plt.xlabel('mesh size')
plt.ylabel('l2 error')

plt.plot(x_data, l2_error_nbc, 'bs', label="l2 error")
plt.plot(x_data, O1_error_nbc, 'r--', label="O(n^1) error", linewidth=1)
plt.plot(x_data, O2_error_nbc, 'g--', label="O(n^2) error", linewidth=1)
plt.legend(loc="upper right")
plt.show()

