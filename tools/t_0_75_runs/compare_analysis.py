"""
Code to compare 2D data of different mesh density
"""
import numpy as np
from scipy.interpolate import griddata
import csv
import matplotlib.pyplot as plt


def read_data(filename):
    points, field = [], []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            points.append([float(row[0]), float(row[1])])
            field.append(float(row[2]))
    print("Reading from file {} successful".format(filename))

    return np.array(points), np.array(field)


def write_data(filename, points, data):
    n_v = data.shape[0]
    with open(filename, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        for i in range(n_v):
            writer.writerow([points[i, 0], points[i, 1], data[i]])
    print("Writing to file {} successful".format(filename))


def write_data_from_dict(filename, data):
    with open(filename, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        for point, value in data.items():
            writer.writerow([point[0], point[1], value])
    print("Writing to file {} successful".format(filename))


def compare_monolithic(res_num, ref_coords, ref_data):
    # Read data of current mesh resolution
    edge_pts, edge_f = read_data('./output_' + str(res_num) + '/parallax_' + str(res_num) + '.csv')
    core_pts, core_f = read_data('./output_' + str(res_num) + '/polar_' + str(res_num) + '.csv')

    # Interpolate fine resolution reference data to current resolution points
    interp_edge_f = griddata(ref_coords, ref_data, edge_pts, method='cubic', fill_value=0.0)
    interp_core_f = griddata(ref_coords, ref_data, core_pts, method='cubic', fill_value=0.0)

    assert interp_edge_f.shape == edge_f.shape
    assert interp_core_f.shape == core_f.shape

    print("---------- Monolithic ----------")
    print("Mesh resolution ({}): Polar result has {} point, PARALLAX result has {} point".format(res_num, core_f.size,
                                                                                                 edge_f.size))

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
    print("--------------------------------")

    return error_edge, error_core


def compare_coupled(res_num, ref_coords, ref_data):
    # Read data of current mesh resolution
    edge_pts, edge_f = read_data('./output_coupling_' + str(res_num) + '/parallax_' + str(res_num) + '.csv')
    core_pts, core_f = read_data('./output_coupling_' + str(res_num) + '/polar_' + str(res_num) + '.csv')

    # Interpolate fine resolution reference data to current resolution points
    interp_edge_f = griddata(ref_coords, ref_data, edge_pts, method='cubic', fill_value=0.0)
    interp_core_f = griddata(ref_coords, ref_data, core_pts, method='cubic', fill_value=0.0)

    assert interp_edge_f.shape == edge_f.shape
    assert interp_core_f.shape == core_f.shape

    print("---------- Coupling ----------")
    print("Mesh resolution ({}): Polar result has {} point, PARALLAX result has {} point".format(res_num, core_f.size,
                                                                                                 edge_f.size))

    # Convert array data to dicts so that reference value can be mapped to points for
    # both Core and Edge participant of coupling
    edge_coupling_f = {tuple(key): value for key, value in zip(edge_pts, edge_f)}
    core_coupling_f = {tuple(key): value for key, value in zip(core_pts, core_f)}
    edge_ref_f = {tuple(key): value for key, value in zip(edge_pts, interp_edge_f)}
    core_ref_f = {tuple(key): value for key, value in zip(core_pts, interp_core_f)}

    coupled_pts = np.concatenate((edge_pts, core_pts), axis=0)
    coupled_vals = np.concatenate((edge_f, core_f), axis=0)

    diff_val = {tuple(key): value for key, value in zip(coupled_pts, coupled_vals)}
    # Set all values of diff_val = 0
    for point in diff_val.keys():
        diff_val[point] = 0

    error_coupling, ref_sum = 0, 0
    error_edge, ref_edge = 0, 0
    # Calculate point wise error for Edge participant
    for point in edge_ref_f.keys():
        if edge_ref_f[point] != 0:
            error_coupling += (edge_coupling_f[point] - edge_ref_f[point]) ** 2
            ref_sum += edge_ref_f[point] ** 2

            error_edge += (edge_coupling_f[point] - edge_ref_f[point]) ** 2
            ref_edge += edge_ref_f[point] ** 2

            diff_val[point] = edge_coupling_f[point] - edge_ref_f[point]

    error_edge = pow(error_edge / ref_edge, 0.5)
    print("Error between reference and Edge participant with mesh res ({}) = {}".format(res, error_edge))

    error_core, ref_core = 0, 0
    for point in core_ref_f.keys():
        # Calculate point wise error for Core participant
        if core_ref_f[point] != 0:
            error_coupling += (core_coupling_f[point] - core_ref_f[point]) ** 2
            ref_sum += core_ref_f[point] ** 2

            error_core += (core_coupling_f[point] - core_ref_f[point]) ** 2
            ref_core += core_ref_f[point] ** 2

            diff_val[point] = core_coupling_f[point] - core_ref_f[point]

    error_core = pow(error_core / ref_core, 0.5)
    print("Error between reference and Core participant with mesh res ({}) = {}".format(res, error_core))

    error_coupling = pow(error_coupling / ref_sum, 0.5)
    print("Error between reference and coupled case with mesh res ({}) = {}".format(res, error_coupling))
    print("------------------------------")

    # write_data_from_dict('./output_coupling_' + str(res_num) + '/diff_val.csv', diff_val)

    return error_coupling


def plot_cross_section(res_num, rmin, rmax_edge, rmin_core, rmax):
    core_x, edge_x = [], []
    core_pts, edge_pts = 667, 333
    dr_core = (rmax_edge - rmin) / core_pts
    dr_edge = (rmax - rmin_core) / edge_pts
    dr = dr_core + dr_edge / 2

    for i in range(core_pts):
        core_x.append([rmin + dr_core*i, 0.0])

    for i in range(edge_pts):
        edge_x.append([rmin_core + dr_edge*i, 0.0])

    core_x = np.array(core_x)
    edge_x = np.array(edge_x)

    # Read data of current mesh resolution
    core_coords, core_f = read_data('./output_coupling_' + str(res_num) + '/polar_' + str(res_num) + '.csv')
    edge_coords, edge_f = read_data('./output_coupling_' + str(res_num) + '/parallax_' + str(res_num) + '.csv')

    # Interpolate fine resolution reference data to current resolution points
    line_core_f = griddata(core_coords, core_f, core_x, method='cubic', fill_value=0.0)
    line_edge_f = griddata(edge_coords, edge_f, edge_x, method='cubic', fill_value=0.0)

    # Gradient for Core and Edge
    gradient_core = np.zeros(core_pts-1)
    for i in range(core_pts-1):
        gradient_core[i] = (line_core_f[i+1] - line_core_f[i])/dr_core

    gradient_edge = np.zeros(edge_pts-1)
    for i in range(edge_pts-1):
        gradient_edge[i] = (line_edge_f[i+1] - line_edge_f[i])/dr_edge

    # Join core and edge data
    coupled_pts = np.concatenate((core_x[:, 0], edge_x[:, 0]), axis=0)
    coupled_vals = np.concatenate((line_core_f, line_edge_f), axis=0)

    gradient_vals = np.zeros(core_pts + edge_pts - 1)
    for i in range(core_pts+edge_pts-1):
        gradient_vals[i] = (coupled_vals[i+1] - coupled_vals[i]) / dr

    return core_x[:, 0], edge_x[:, 0], line_core_f, line_edge_f, gradient_core, gradient_edge


def plot_ref_cross_section(ref_f, ref_pts):
    line_x = []
    n_points = 1000
    dr = (0.5 - 0.2) / n_points

    for i in range(n_points):
        line_x.append([0.2 + dr * i, 0.0])

    line_pts = np.array(line_x)

    line_ref_f = griddata(ref_pts, ref_f, line_pts, method='cubic', fill_value=0.0)

    gradient_vals = np.zeros(n_points - 1)
    for i in range(n_points - 1):
        gradient_vals[i] = (line_ref_f[i+1] - line_ref_f[i]) / dr

    return line_pts[:, 0], line_ref_f, gradient_vals


# Finest resolution for Polar code considered as reference result
ref_points, ref_field = read_data('./output_ref/polar_ref2.csv')
print("Reference result has {} points".format(ref_field.size))

# For each mesh resolution, fit data by interpolation and calculate error by L2 norm
mesh_res = 4

rmin_core = 0.2
rmax_core = 0.4
rmin_edge = [0.388, 0.394, 0.397, 0.3985]
rmax_edge = 0.5


corepts_plot, edgepts_plot = [], []
corev_plot, edgev_plot, core_flux_plot, edge_flux_plot = [], [], [], []
core_pts, edge_pts = None, None

for res in range(mesh_res):
    print("Getting line plot for mesh resolution number: {}".format(res))
    core_pts, edge_pts, corev, edgev, core_flux, edge_flux = plot_cross_section(res, rmin_core, rmax_core, rmin_edge[res],
                                                                                rmax_edge)
    corepts_plot.append(core_pts)
    edgepts_plot.append(edge_pts)
    corev_plot.append(corev)
    edgev_plot.append(edgev)
    core_flux_plot.append(core_flux)
    edge_flux_plot.append(edge_flux)

refpts_plot, refv_plot, ref_flux_plot = plot_ref_cross_section(ref_field, ref_points)

# plot
plt.xlabel('x coordinate')
plt.ylabel('field value')
plt.title('Values along line section, t = 0.75')

plt.plot(refpts_plot, refv_plot, 'k-', label="Reference Result", linewidth=1.5)

plt.plot(corepts_plot[0], corev_plot[0], 'bo', label="Core Mesh Res 0")
plt.plot(edgepts_plot[0], edgev_plot[0], 'ro', label="Edge Mesh Res 0")

plt.plot(corepts_plot[1], corev_plot[1], 'b--', label="Core Mesh Res 1")
plt.plot(edgepts_plot[1], edgev_plot[1], 'r--', label="Edge Mesh Res 1")

plt.plot(corepts_plot[2], corev_plot[2], 'b-*', label="Core Mesh Res 2")
plt.plot(edgepts_plot[2], edgev_plot[2], 'r-*', label="Edge Mesh Res 2")

plt.plot(corepts_plot[3], corev_plot[3], 'b-o', label="Core Mesh Res 3")
plt.plot(edgepts_plot[3], edgev_plot[3], 'r-o', label="Edge Mesh Res 3")

plt.legend(loc='best')
plt.show()

# Remove last x coordinate as gradient by forward difference is not computed at this point
core_flux_pts = []
edge_flux_pts = []
for res in range(mesh_res):
    corepts = []
    for i in range(666):
        corepts.append(corepts_plot[res][i])
    core_flux_pts.append(corepts)

    edgepts = []
    for i in range(332):
        edgepts.append(edgepts_plot[res][i])
    edge_flux_pts.append(edgepts)

ref_flux_pts = []
for i in range(999):
    ref_flux_pts.append(refpts_plot[i])

plt.xlabel('x coordinate')
plt.ylabel('flux')
plt.title('Flux along line section, t = 0.75')

plt.plot(ref_flux_pts, ref_flux_plot, 'k-', label="Reference Result", linewidth=1.5)

plt.plot(core_flux_pts[0], core_flux_plot[0], 'bo', label="Core Mesh Res 0")
plt.plot(edge_flux_pts[0], edge_flux_plot[0], 'ro', label="Edge Mesh Res 0")

plt.plot(core_flux_pts[1], core_flux_plot[1], 'b--', label="Core Mesh Res 1")
plt.plot(edge_flux_pts[1], edge_flux_plot[1], 'r--', label="Edge Mesh Res 1")

plt.plot(core_flux_pts[2], core_flux_plot[2], 'b-*', label="Core Mesh Res 2")
plt.plot(edge_flux_pts[2], edge_flux_plot[2], 'r-*', label="Edge Mesh Res 2")

plt.plot(core_flux_pts[3], core_flux_plot[3], 'b-o', label="Core Mesh Res 3")
plt.plot(edge_flux_pts[3], edge_flux_plot[3], 'r-o', label="Edge Mesh Res 3")

plt.legend(loc='best')
plt.show()

err_edge = np.zeros(mesh_res)
err_core = np.zeros(mesh_res)
err_coupling = np.zeros(mesh_res)
for res in range(mesh_res):
    print("Comparing mesh resolution number: {}".format(res))
    # err_edge[res], err_core[res] = compare_monolithic(res, ref_points, ref_field)
    err_coupling[res] = compare_coupled(res, ref_points, ref_field)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('mesh size')
plt.ylabel('l2 error')
plt.title('Overlap 2*dx, t = 0.75')

mesh_resolutions = [50, 100, 200, 400]
# # Plot monolithic PARALLAX comparison
# plt.plot(mesh_resolutions, err_edge, 'rs', label="Edge monolithic", linewidth=2)
# O1_err_edge = [err_edge[0], err_edge[0]/2, err_edge[0]/4]
# plt.plot(mesh_resolutions, O1_err_edge, 'r--', label="O(1) Edge", linewidth=1)
# O2_err_edge = [err_edge[0], err_edge[0]/4, err_edge[0]/16]
# plt.plot(mesh_resolutions, O2_err_edge, 'r-.', label="O(2) Edge", linewidth=1)
#
# # Plot monolithic Polar code comparison
# plt.plot(mesh_resolutions, err_core, 'gs', label="Core monolithic", linewidth=2)
# O1_err_core = [err_core[0], err_core[0]/2, err_core[0]/4]
# plt.plot(mesh_resolutions, O1_err_core, 'g--', label="O(1) Core", linewidth=1)
# O2_err_core = [err_core[0], err_core[0]/4, err_core[0]/16]
# plt.plot(mesh_resolutions, O2_err_core, 'g-.', label="O(2) Core", linewidth=1)

# Plot PARALLAX - Polar coupled solution comparison
plt.plot(mesh_resolutions, err_coupling, 'bs', label="Coupling", linewidth=2)
O1_err_coupling = [err_coupling[0], err_coupling[0]/2, err_coupling[0]/4, err_coupling[0]/8]
plt.plot(mesh_resolutions, O1_err_coupling, 'b--', label="O(1) Coupling", linewidth=1)
O2_err_coupling = [err_coupling[0], err_coupling[0]/4, err_coupling[0]/16, err_coupling[0]/64]
plt.plot(mesh_resolutions, O2_err_coupling, 'b-.', label="O(2) Coupling", linewidth=1)

plt.legend(loc='best')
plt.show()
