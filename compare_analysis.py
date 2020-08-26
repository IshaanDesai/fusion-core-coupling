"""
Code to compare 2D data of different mesh density
"""
import numpy as np
from scipy.interpolate import griddata
import csv
import matplotlib.pyplot as plt


def read_data(res, code_name, n_t, n_displ):
    # Read data from CSV files of specified code
    points, field = [], []
    for n in range(n_t + n_displ, n_displ):
        with open('./output_' + str(res) + '/' + code_name + '_' + str(n) + '.csv', 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                points.append([row[0], row[1]])
                field.append(row[2])

    return np.array(points), np.array(field)


# Finest resolution (3) is reference result
ref_points, ref_field = read_data(3, 'polar', 10000, 1000)

# For each mesh resolution, fit data by interpolation and calculate error by L2 norm
mesh_res = 3
err_edge = np.zeros(mesh_res)
err_core = np.zeros(mesh_res)
for res in range(mesh_res):
    edge_pts, edge_f = read_data(res, 'parallax', 10000, 1000)
    core_pts, core_f = read_data(res, 'polar', 10000, 1000)
    interp_edge_f = griddata(ref_points, ref_field, edge_pts, method='cubic')
    interp_core_f = griddata(ref_points, ref_field, core_pts, method='cubic')

    assert interp_edge_f.shape == ref_field.shape
    assert interp_core_f.shape == ref_field.shape

    i = 0
    for u_ref, u_edge in interp_edge_f, ref_field:
        err_edge[res] += abs(u_edge[i] - u_ref[i])/abs(u_ref[i])
        i += 1

    print("Error between reference and edge case with mesh res ({}) = {}".format(res, err_edge[res]))

    i = 0
    for u_ref, u_core in interp_edge_f, ref_field:
        err_core[res] += abs(u_edge[i] - u_ref[i]) / abs(u_ref[i])
        i += 1

    print("Error between reference and core case with mesh res ({}) = {}".format(res, err_core[res]))

# Plotting
plt.xscale('log')
plt.yscale('log')
plt.xlabel('mesh size')
plt.ylabel('l2 error')

mesh_resolutions = [10, 100, 1000]
O1_err_edge = [err_edge[0], err_edge[0]/2, err_edge[0]/4]
O2_err_edge = [err_edge[0], err_edge[0]/4, err_edge[0]/16]
O1_err_core = [err_core[0], err_core[0]/2, err_core[0]/4]
O2_err_core = [err_core[0], err_core[0]/4, err_core[0]/16]
plt.plot(mesh_resolutions, err_edge, 'r.', label="Edge monolithic")
plt.plot(mesh_resolutions, O1_err_edge, 'r--', label="O(1) Edge", linewidth=1)
plt.plot(mesh_resolutions, O2_err_edge, 'r-.', label="O(2) Edge", linewidth=1)
plt.plot(mesh_resolutions, err_core, 'g.', label="Core monolithic")
plt.plot(mesh_resolutions, O1_err_core, 'g--', label="O(1) Core", linewidth=1)
plt.plot(mesh_resolutions, O2_err_core, 'g-.', label="O(2) Core", linewidth=1)
plt.legend(loc="upper right")
plt.show()

