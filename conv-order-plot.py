"""
Convergence plot for error of numerical simulation with analytical solution derived from Bessel functions
"""
import matplotlib.pyplot as plt

def calculate_baselines(l2, linf):
    base_val = (l2[0] + linf[0]) / 2
    n1 = [base_val, base_val / 2, base_val / 4, base_val / 8, base_val / 16]
    n2 = [base_val, base_val / 4, base_val / 16, base_val / 64, base_val / 256]

    return n1, n2

def plot_graph(title, n1, n2, l2, linf):
    # Number of grid points in radial / x- direction
    dx = [3.2e-2, 1.6e-2, 8e-3, 4e-3, 2e-3]
    xi = list(range(len(dx)))

    # Common plot settings
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('dx')
    plt.ylabel('Error')

    plt.title(title)

    plt.plot(xi, l2, 'rs-', label="l2 error")
    plt.plot(xi, linf, 'bs-', label="linf error")
    plt.plot(xi, n1, 'k-*', label="O(n^1) error")
    plt.plot(xi, n2, 'k--', label="O(n^2) error")
    plt.xticks(xi, dx)
    plt.legend(loc="upper right")
    plt.yscale('log')
    plt.xlabel('dx')
    plt.ylabel('Error')
    plt.show()


# ---- Variable overlap and variable support radius ----
# Error of mapping flux from Edge to Core
l2_core = [1.479e-3, 4.434e-4, 1.287e-4, 3.680e-5, 1.461e-5]
linf_core = [2.553e-3, 7.671e-4, 2.440e-4, 7.280e-5, 2.896e-5]

n1_core, n2_core = calculate_baselines(l2_core, linf_core)
plot_graph('Mapping Cartesian -> Polar \n variable overlap and variable support radius', n1_core, n2_core, l2_core, linf_core)

# Error of mapping values from Core to Edge
l2_edge = [3.674e-4, 6.176e-5, 2.476e-5, 1.195e-5, 7.141e-6]
linf_edge = [6.655e-4, 1.624e-4, 6.689e-5, 3.981e-5, 2.580e-5]

n1_edge, n2_edge = calculate_baselines(l2_edge, linf_edge)
plot_graph('Mapping Polar -> Cartesian \n variable overlap and variable support radius', n1_edge, n2_edge, l2_edge, linf_edge)
# ------------------------------------------------------

# ---- Constant overlap and variable support radius ----
# Error of mapping flux from Edge to Core
l2_core = [1.479e-3, 3.604e-4, 1.177e-4, 4.386e-5, 1.897e-5]
linf_core = [2.553e-3, 7.211e-4, 3.002e-4, 1.209e-4, 5.343e-5]

n1_core, n2_core = calculate_baselines(l2_core, linf_core)
plot_graph('Mapping Cartesian -> Core \n constant overlap and variable support radius', n1_core, n2_core, l2_core, linf_core)

# Error of mapping values from Core to Edge
l2_edge = [3.458e-4, 9.932e-5, 3.393e-5, 1.447e-5, 6.602e-6]
linf_edge = [6.357e-4, 2.104e-4, 8.119e-5, 3.504e-5, 1.604e-5]

n1_edge, n2_edge = calculate_baselines(l2_edge, linf_edge)
plot_graph('Mapping Polar -> Cartesian \n constant overlap and variable support radius', n1_edge, n2_edge, l2_edge, linf_edge)
# ------------------------------------------------------

# ---- Variable overlap and global RBF functions ----
# Error of mapping flux from Edge to Core
l2_core = [1.719e-3, 5.282e-4, 1.397e-4, 0.0, 0.0]
linf_core = [2.434e-3, 7.497e-4, 1.987e-4, 0.0, 0.0]

n1_core, n2_core = calculate_baselines(l2_core, linf_core)
plot_graph('Mapping Cartesian -> Core \n constant overlap and global RBFs', n1_core, n2_core, l2_core, linf_core)

# Error of mapping values from Core to Edge
l2_edge = [2.666e-4, 3.699e-5, 5.425e-6, 0.0, 0.0]
linf_edge = [4.176e-4, 5.894e-5, 8.301e-6, 0.0, 0.0]

n1_edge, n2_edge = calculate_baselines(l2_edge, linf_edge)
plot_graph('Mapping Polar -> Cartesian \n constant overlap and global RBFs', n1_edge, n2_edge, l2_edge, linf_edge)
# ------------------------------------------------------
