"""
Convergence plot for error of numerical simulation with analytical solution derived from Bessel functions
"""
import matplotlib.pyplot as plt

dx = [3.2e-2, 1.6e-2, 8e-3, 4e-3, 2e-3, 1e-3, 5e-4]


def calculate_baselines(l2, linf):
    base_val = (l2[0] + linf[0]) / 2  # Take the midpoint of highest l2 and l_inf error as starting point for baseline
    n1 = [base_val]
    n2 = [base_val]
    for i in range(1, len(dx)):
        n1.append(base_val / (i * 2))
        n2.append(base_val / (i * 4))

    return n1, n2


def plot_graph(title, l2, linf):
    # Calculate first order and second order baseline plot lines
    n1, n2 = calculate_baselines(l2, linf)

    xi = list(range(len(dx)))

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


case = 'Overlap 5 layers\nRBF mapping tps with compact support'

mapping_direction = 'Cartesian -> Polar'
l2_data = [4.101E-04, 1.476E-04, 4.980E-05, 2.282E-05, 1.256E-05, 5.555E-06, None]
linf_data = [8.781E-04, 2.773E-04, 1.148E-04, 4.967E-05, 2.794E-05, 1.376E-05, None]
plot_graph(case + '\n' + mapping_direction, l2_data, linf_data)

mapping_direction = 'Polar -> Cartesian'
l2_data = [1.287E-04, 4.899E-05, 2.393E-05, 1.033E-05, 7.171E-06, 3.788E-06, None]
linf_data = [2.837E-04, 1.241E-04, 6.859E-05, 2.827E-05, 2.602E-05, 1.468E-05, None]
plot_graph(case + '\n ' + mapping_direction, l2_data, linf_data)

case = 'Overlap 5 layers\nRBF mapping tps with global support solved with QR decomposition'

mapping_direction = 'Cartesian -> Polar'
l2_data = [3.566E-05, 3.114E-06, 3.240E-07, None, None, None, None]
linf_data = [6.823E-05, 5.098E-06, 6.596E-07, None, None, None, None]
plot_graph(case + '\n ' + mapping_direction, l2_data, linf_data)

mapping_direction = 'Polar -> Cartesian'
l2_data = [9.028E-06, 2.985E-07, 3.517E-07, 1.819E-08, None, None, None]
linf_data = [1.827E-05, 1.235E-06, 1.490E-06, 8.258E-08, None, None, None]
plot_graph(case + '\n ' + mapping_direction, l2_data, linf_data)

case = 'Overlap 5 layers\nRBF mapping tps with global support solved with PETSc GMRES'

mapping_direction = 'Cartesian -> Polar'
l2_data = [3.577E-05, 3.114E-06, 1.153E-06, None, None, None, None]
linf_data = [6.818E-05, 5.098E-06, 4.441E-06, None, None, None, None]
plot_graph(case + '\n ' + mapping_direction, l2_data, linf_data)

mapping_direction = 'Polar -> Cartesian'
l2_data = [9.136E-06, 1.313E-06, 3.517E-07, None, None, None, None]
linf_data = [1.864E-05, 2.895E-06, 1.490E-06, None, None, None, None]
plot_graph(case + '\n ' + mapping_direction, l2_data, linf_data)

case = 'Overlap one layer\nRBF mapping tps with compact support'

mapping_direction = 'Cartesian -> Polar'
l2_data = [7.247E-03, 2.210E-03, 8.441E-04, 3.717E-04, 1.718E-04, 8.320E-05, None]
linf_data = [2.229E-02, 5.122E-03, 2.829E-03, 1.103E-03, 5.337E-04, 2.628E-04, None]
plot_graph(case + '\n' + mapping_direction, l2_data, linf_data)

mapping_direction = 'Polar -> Cartesian'
l2_data = [9.536E-04, 1.101E-03, 6.209E-04, 3.224E-04, 1.669E-04, 8.027E-05, None]
linf_data = [4.149E-03, 4.967E-03, 2.947E-03, 1.654E-03, 8.519E-04, 4.124E-04, None]
plot_graph(case + '\n ' + mapping_direction, l2_data, linf_data)

case = 'Overlap one layer\nRBF mapping tps with global support solved with QR decomposition'

mapping_direction = 'Cartesian -> Polar'
l2_data = [2.965E-03, 8.113E-04, 2.021E-04, 5.398E-05, 1.312E-05, None, None]
linf_data = [6.267E-03, 2.119E-03, 5.363E-04, 1.384E-04, 3.363E-05, None, None]
plot_graph(case + '\n ' + mapping_direction, l2_data, linf_data)

mapping_direction = 'Polar -> Cartesian'
l2_data = [4.897E-04, 2.227E-04, 6.417E-05, 1.705E-05, 4.400E-06, None, None]
linf_data = [1.260E-03, 9.951E-04, 3.018E-04, 8.727E-05, 2.245E-05, None, None]
plot_graph(case + '\n ' + mapping_direction, l2_data, linf_data)

case = 'Overlap one layer\nRBF mapping tps with global support solved with PETSc GMRES'

mapping_direction = 'Cartesian -> Polar'
l2_data = [3.360E-03, 9.183E-04, 2.291E-04, 5.925E-05, 2.919E-05, None, None]
linf_data = [7.136E-03, 2.386E-03, 6.055E-04, 1.426E-04, 5.317E-05, None, None]
plot_graph(case + '\n ' + mapping_direction, l2_data, linf_data)

mapping_direction = 'Polar -> Cartesian'
l2_data = [4.522E-04, 2.187E-04, 6.347E-05, 1.691E-05, 4.370E-06, None, None]
linf_data = [1.864E-05, 2.895E-06, 1.490E-06, None, None, None, None]
plot_graph(case + '\n ' + mapping_direction, l2_data, linf_data)