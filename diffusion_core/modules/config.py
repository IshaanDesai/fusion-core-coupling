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
        self._read_mesh_name = None
        self._write_mesh_name = None
        self._read_data_name = None
        self._write_data_name = None

        self._meshtype = None

        self._diffc = None

        self._coupling_on = None
        self._dt = None
        self._t_total = None
        self._t_out = None

        self._m = None
        self._ums = None

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
        self._read_mesh_name = data["interface"]["read_mesh_name"]
        self._write_mesh_name = data["interface"]["write_mesh_name"]
        self._write_data_name = data["interface"]["write_data_name"]
        self._read_data_name = data["interface"]["read_data_name"]

        self._meshtype = data["mesh_parameters"]["type"]

        self._diffc = data["diffusion_parameters"]["diff_coeff"]

        self._coupling_on = data["simulation_parameters"]["coupling_on"]
        self._dt = data["simulation_parameters"]["timestep"]
        self._t_total = data["simulation_parameters"]["total_time"]
        self._t_out = data["simulation_parameters"]["t_output"]

        self._m = data["bessel_params"]["pol_mode_number"]
        self._ums = data["bessel_params"]["ums"]

        read_file.close()

    def get_config_file_name(self):
        return self._config_file_name

    def get_participant_name(self):
        return self._participant_name

    def get_read_mesh_name(self):
        return self._read_mesh_name

    def get_write_mesh_name(self):
        return self._write_mesh_name

    def get_read_data_name(self):
        return self._read_data_name

    def get_write_data_name(self):
        return self._write_data_name

    def get_mesh_type(self):
        return self._meshtype

    def get_diffusion_coeff(self):
        return self._diffc

    def get_dt(self):
        return self._dt

    def get_total_time(self):
        return self._t_total

    def get_t_output(self):
        return self._t_out

    def get_pol_mode_number(self):
        return self._m

    def get_ums(self):
        return self._ums

    def is_coupling_on(self):
        return self._coupling_on