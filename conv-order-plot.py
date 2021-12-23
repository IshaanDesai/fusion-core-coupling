"""
Convergence plot for error of numerical simulation with analytical solution derived from Bessel functions
"""
import matplotlib.pyplot as plt

# Common plot settings
plt.legend(loc="upper right")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('mesh dx')
plt.ylabel('Error')

# Number of grid points in radial / x- direction
dx = [4e-3, 8e-3, 1.6e-2, 3.2e-2]

# Variable overlap and variable support radius
# Error of anayltical flux mapping from Edge to Core
l2_core = [2.282e-5, 4.980e-5, 1.476e-4, 4.097e-4]
linf_core = [4.967e-5, 1.148e-4, 2.770e-4, 8.771e-4]

# Error of analytical values mapping from Core to Edge
l2_edge = [1.195e-5, 2.476e-5, 6.176e-5, 3.674e-4]
linf_edge = [3.981e-5, 6.689e-5, 1.624e-4, 6.655e-4]

plt.title('Error of flux mapping from Edge to Core: variable overlap and variable support radius')

plt.plot(dx, l2_core, 'rs', label="l2 error")
plt.plot(dx, linf_core, 'bs', label="linf error")
plt.show()
