"""
Code to compare 2D data of different mesh density
"""
import numpy as np
from scipy.interpolate import griddata
import csv
import matplotlib.pyplot as plt


def read_data(res_str, code_name, n_t):
    points, field = [], []
    filename = './output_' + res_str + '/' + code_name + '_' + str(n_t) + '.csv'
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            points.append([float(row[0]), float(row[1])])
            field.append(float(row[2]))
    print("Reading from file {} successful".format(filename))

    return np.array(points), np.array(field)


def write_data(res_str, code_name, points, data):
    filename = './output_' + res_str + '/' + code_name + '_new.csv'
    n_v = data.shape[0]
    with open(filename, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        for i in range(n_v):
            writer.writerow([points[i, 0], points[i, 1], data[i]])
    print("Writing to file {} successful".format(filename))


def compare_monolithic(mesh_res, tstamp, ref_points, ref_data):
    edge_pts, edge_f = read_data(str(mesh_res), 'parallax', tstamp)
    core_pts, core_f = read_data(str(mesh_res), 'polar', tstamp)
    interp_edge_f = griddata(ref_points, ref_data, edge_pts, method='cubic', fill_value=0.0)
    interp_core_f = griddata(ref_points, ref_data, core_pts, method='cubic', fill_value=0.0)

    assert interp_edge_f.shape == edge_f.shape
    assert interp_core_f.shape == core_f.shape

    print("---------- Monolithic ----------")
    print("Mesh resolution ({}): Polar result has {} point, PARALLAX result has {} point".format(mesh_res, core_f.size,
                                                                                                 edge_f.size))

    # write_data(str(mesh_res), 'edge', edge_pts, interp_edge_f)
    # write_data(str(mesh_res), 'core', core_pts, interp_core_f)

    error_edge, ref_sum = 0, 0
    for i in range(edge_f.size):
        u_ref = interp_edge_f[i]
        if u_ref != 0:
            error_edge += (edge_f[i] - u_ref) ** 2
            ref_sum += u_ref ** 2

    error_edge = pow(error_edge / ref_sum, 0.5)
    print("Error between reference and edge case with mesh res ({}) = {}".format(res, error_edge))

    error_core, ref_sum = 0, 0
    for i in range(core_f.size):
        u_ref = interp_core_f[i]
        if u_ref != 0:
            error_core += (core_f[i] - u_ref) ** 2
            ref_sum += u_ref ** 2

    error_core = pow(error_core / ref_sum, 0.5)
    print("Error between reference and core case with mesh res ({}) = {}".format(res, error_core))
    print("--------------------")

    return error_edge, error_core


def compare_coupled(mesh_res, tstamp, ref_points, ref_data):
    edge_pts, edge_f = read_data(str(mesh_res), 'parallax', tstamp)
    core_pts, core_f = read_data(str(mesh_res), 'polar', tstamp)
    interp_edge_f = griddata(ref_points, ref_data, edge_pts, method='cubic', fill_value=0.0)
    interp_core_f = griddata(ref_points, ref_data, core_pts, method='cubic', fill_value=0.0)

    assert interp_edge_f.shape == edge_f.shape
    assert interp_core_f.shape == core_f.shape

    print("---------- Coupling ----------")
    print("Mesh resolution ({}): Polar result has {} point, PARALLAX result has {} point".format(mesh_res, core_f.size,
                                                                                                 edge_f.size))

    # Convert array data to dicts so that reference value can be mapped to points for
    # both Core and Edge participant of coupling
    edge_coupling_f = {tuple(key): value for key, value in zip(edge_pts, edge_f)}
    core_coupling_f = {tuple(key): value for key, value in zip(core_pts, core_f)}
    edge_ref_f = {tuple(key): value for key, value in zip(edge_pts, interp_edge_f)}
    core_ref_f = {tuple(key): value for key, value in zip(core_pts, interp_core_f)}

    error_edge, ref_sum = 0, 0
    for point in edge_ref_f.keys():
        if edge_ref_f[point] != 0:
            error_edge += (edge_coupling_f[point] - edge_ref_f[point]) ** 2
            ref_sum += edge_ref_f[point] ** 2

    error_edge = pow(error_edge / ref_sum, 0.5)
    print("Error between reference and edge case with mesh res ({}) = {}".format(res, error_edge))

    error_core, ref_sum = 0, 0
    for point in core_ref_f.keys():
        if core_ref_f[point] != 0:
            error_core += (core_coupling_f[point] - core_ref_f[point]) ** 2
            ref_sum += core_ref_f[point] ** 2

    error_core = pow(error_core / ref_sum, 0.5)
    print("Error between reference and core case with mesh res ({}) = {}".format(res, error_core))
    print("--------------------")

    return error_edge, error_core


# Finest resolution for Polar code considered as reference result
ref_points, ref_field = read_data('ref', 'polar', 64101)
print("Reference result has {} points".format(ref_field.size))

# For each mesh resolution, fit data by interpolation and calculate error by L2 norm
mesh_res = 3

err_edge = np.zeros(mesh_res)
err_core = np.zeros(mesh_res)
err_coupling = np.zeros(mesh_res)
for res in range(mesh_res):
    print("Comparing mesh resolution number: {}".format(res))
    err_edge[res], err_core[res] = compare_monolithic(res, res, ref_points, ref_field)
    err_coupling_1, err_coupling_2 = compare_coupled('coupling_'+str(res), res, ref_points, ref_field)
    err_coupling[res] = err_coupling_1 + err_coupling_2

# Plotting
plt.xscale('log')
plt.yscale('log')
plt.xlabel('mesh size')
plt.ylabel('l2 error')

mesh_resolutions = [50, 100, 200]
plt.plot(mesh_resolutions, err_edge, 'rs', label="Edge monolithic", linewidth=2)
O1_err_edge = [err_edge[0], err_edge[0]/2, err_edge[0]/4]
plt.plot(mesh_resolutions, O1_err_edge, 'r--', label="O(1) Edge", linewidth=1)
O2_err_edge = [err_edge[0], err_edge[0]/4, err_edge[0]/16]
plt.plot(mesh_resolutions, O2_err_edge, 'r-.', label="O(2) Edge", linewidth=1)

plt.plot(mesh_resolutions, err_core, 'gs', label="Core monolithic", linewidth=2)
O1_err_core = [err_core[0], err_core[0]/2, err_core[0]/4]
plt.plot(mesh_resolutions, O1_err_core, 'g--', label="O(1) Core", linewidth=1)
O2_err_core = [err_core[0], err_core[0]/4, err_core[0]/16]
plt.plot(mesh_resolutions, O2_err_core, 'g-.', label="O(2) Core", linewidth=1)

plt.plot(mesh_resolutions, err_coupling, 'bs', label="Coupling", linewidth=2)
O1_err_coupling = [err_coupling[0], err_coupling[0]/2, err_coupling[0]/4]
plt.plot(mesh_resolutions, O1_err_coupling, 'b--', label="O(1) Coupling", linewidth=1)
O2_err_coupling = [err_coupling[0], err_coupling[0]/4, err_coupling[0]/16]
plt.plot(mesh_resolutions, O2_err_coupling, 'b-.', label="O(2) Coupling", linewidth=1)

plt.legend(loc='best')
plt.show()

