"""
Code to compare 2D data of different mesh density
"""
import numpy as np
from scipy.interpolate import griddata
import csv
import matplotlib.pyplot as plt


def read_data(res_str, code_name, n_t):
    # Read data from CSV files of specified code
    points, field = [], []
    with open('./output_' + res_str + '/' + code_name + '_' + str(n_t) + '.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            points.append([row[0], row[1]])
            field.append(row[2])

    return np.array(points), np.array(field)


def compare_data_2d(mesh_res, edge_ts, core_ts, ref_points, ref_data):
    edge_pts, edge_f = read_data(str(mesh_res), 'parallax', edge_ts)
    core_pts, core_f = read_data(str(mesh_res), 'polar', core_ts)
    interp_edge_f = griddata(ref_points, ref_data, edge_pts, method='cubic')
    interp_core_f = griddata(ref_points, ref_data, core_pts, method='cubic')

    assert interp_edge_f.shape == ref_field.shape
    assert interp_core_f.shape == ref_field.shape

    i, error_edge = 0, 0
    for u_edge, u_ref in edge_f, interp_edge_f:
        error_edge += abs(u_edge[i] - u_ref[i]) / abs(u_ref[i])
        i += 1

    print("Error between reference and edge case with mesh res ({}) = {}".format(res, err_edge[res]))

    i, error_core = 0, 0
    for u_core, u_ref in core_f, interp_edge_f:
        error_core += abs(u_core[i] - u_ref[i]) / abs(u_ref[i])
        i += 1

    print("Error between reference and core case with mesh res ({}) = {}".format(res, err_core[res]))

    return error_edge, error_core


# Finest resolution (3) is reference result
ref_points, ref_field = read_data('ref', 'polar', 63999)

# For each mesh resolution, fit data by interpolation and calculate error by L2 norm
mesh_res = 3
# Time stamps for each mesh resolutions in PARALLAX and Polar Code to match names of output files
parallax_ts = [10000, 10000, 10000]
polar_ts = [40000, 50000, 63999]

err_edge = np.zeros(mesh_res)
err_core = np.zeros(mesh_res)
err_coupling = np.zeros(mesh_res)
for res in range(mesh_res):
    err_edge[res], err_core[res] = compare_data_2d(res, parallax_ts[res], polar_ts[res], ref_points, ref_field)
    err_coupling_1, err_coupling_2 = compare_data_2d('coupling_' +
                                                     str(res), parallax_ts[res], polar_ts[res], ref_points, ref_field)
    err_coupling[res] = err_coupling_1 + err_coupling_2

# Plotting
plt.xscale('log')
plt.yscale('log')
plt.xlabel('mesh size')
plt.ylabel('l2 error')

mesh_resolutions = [10, 100, 1000]
plt.plot(mesh_resolutions, err_edge, 'r.', label="Edge monolithic")
O1_err_edge = [err_edge[0], err_edge[0] / 2, err_edge[0] / 4]
plt.plot(mesh_resolutions, O1_err_edge, 'r--', label="O(1) Edge", linewidth=1)
O2_err_edge = [err_edge[0], err_edge[0] / 4, err_edge[0] / 16]
plt.plot(mesh_resolutions, O2_err_edge, 'r-.', label="O(2) Edge", linewidth=1)

plt.plot(mesh_resolutions, err_core, 'g.', label="Core monolithic")
O1_err_core = [err_core[0], err_core[0] / 2, err_core[0] / 4]
plt.plot(mesh_resolutions, O1_err_core, 'g--', label="O(1) Core", linewidth=1)
O2_err_core = [err_core[0], err_core[0] / 4, err_core[0] / 16]
plt.plot(mesh_resolutions, O2_err_core, 'g-.', label="O(2) Core", linewidth=1)

plt.plot(mesh_resolutions, err_coupling, 'b.', label="Coupling")
O1_err_coupling = [err_coupling[0], err_coupling[0] / 2, err_coupling[0] / 4]
plt.plot(mesh_resolutions, O1_err_coupling, 'b--', label="O(1) Coupling", linewidth=1)
O2_err_coupling = [err_coupling[0], err_coupling[0] / 4, err_coupling[0] / 16]
plt.plot(mesh_resolutions, O2_err_coupling, 'b-.', label="O(2) Coupling", linewidth=1)

plt.legend(loc="upper right")
plt.show()
