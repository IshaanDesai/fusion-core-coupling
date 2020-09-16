"""
This is the configuration module of fenicsadapter
"""

import json
import os
import sys


class Config:
    """
    Handles the reading of parameters in the JSON configuration file provided by the user. This class is based on
    the config class in https://github.com/precice/fenics-adapter/tree/develop/fenicsadapter

    :ivar _config_file_name: name of the preCICE configuration file
    :ivar _coupling_mesh_name: name of mesh as defined in preCICE config
    :ivar _read_data_name: name of read data as defined in preCICE config
    :ivar _write_data_name: name of write data as defined in preCICE config
    """

    def __init__(self, config_filename):

        self._config_file_name = None
        self._participant_name = None
        self._coupling_read_mesh_name = None
        self._coupling_write_mesh_name = None
        self._read_data_name = None
        self._write_data_name = None

        self._dims = None
        self._rmin = None
        self._rmax = None
        self._rcustom = None
        self._r_points = None
        self._theta_points = None

        self._diffc_perp = None
        self._dt = None
        self._t_total = None
        self._t_out = None

        self._xb = None
        self._yb = None
        self._wxb = None
        self._wyb = None

        self.read_json(config_filename)

    def read_json(self, config_filename):
        """
        Reads JSON adapter configuration file and saves the data to the respective instance attributes.

        :var path: stores path to the JSON config file
        :var data: data decoded from JSON files
        :var read_file: stores file path
        """
        folder = os.path.dirname(os.path.join(os.getcwd(), os.path.dirname(sys.argv[0]), config_filename))
        path = os.path.join(folder, os.path.basename(config_filename))
        read_file = open(path, "r")
        data = json.load(read_file)
        self._config_file_name = os.path.join(folder, data["config_file_name"])
        self._participant_name = data["participant_name"]
        self._coupling_read_mesh_name = data["interface"]["coupling_read_mesh_name"]
        self._coupling_write_mesh_name = data["interface"]["coupling_write_mesh_name"]
        try:
            self._write_data_name = data["interface"]["write_data_name"]
        except:
            self._write_data_name = None
        self._read_data_name = data["interface"]["read_data_name"]

        self._dims = data["mesh_parameters"]["dimensions"]
        self._rmin = data["mesh_parameters"]["r_inner"]
        self._rmax = data["mesh_parameters"]["r_outer"]
        self._rcustom = data["mesh_parameters"]["r_custom"]
        self._r_points = data["mesh_parameters"]["radial_points"]
        self._theta_points = data["mesh_parameters"]["circular_points"]

        self._diffc_perp = data["diffusion_parameters"]["coeff_perp"]
        self._dt = data["simulation_parameters"]["timestep"]
        self._t_total = data["simulation_parameters"]["total_time"]
        self._t_out = data["simulation_parameters"]["t_output"]

        self._xb = data["init_conditions"]["xb"]
        self._yb = data["init_conditions"]["yb"]
        self._wxb = data["init_conditions"]["wxb"]
        self._wyb = data["init_conditions"]["wyb"]

        read_file.close()

    def get_config_file_name(self):
        return self._config_file_name

    def get_participant_name(self):
        return self._participant_name

    def get_coupling_read_mesh_name(self):
        return self._coupling_read_mesh_name

    def get_coupling_write_mesh_name(self):
        return self._coupling_write_mesh_name

    def get_read_data_name(self):
        return self._read_data_name

    def get_write_data_name(self):
        return self._write_data_name

    def get_dims(self):
        return self._dims

    def get_rmin(self):
        return self._rmin

    def get_rmax(self):
        return self._rmax

    def get_rcustom(self):
        return self._rcustom

    def get_r_points(self):
        return self._r_points

    def get_theta_points(self):
        return self._theta_points

    def get_diffusion_coeff(self):
        return self._diffc_perp

    def get_dt(self):
        return self._dt

    def get_total_time(self):
        return self._t_total

    def get_t_output(self):
        return self._t_out

    def get_xb_yb(self):
        return self._xb, self._yb

    def get_wxb_wyb(self):
        return self._wxb, self._wyb

