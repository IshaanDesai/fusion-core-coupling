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
        self._coupling_mesh_name = None
        self._read_data_name = None
        self._write_data_name = None

        self._dims = None
        self._rhomin = None
        self._rhomax = None
        self._r_spacing = None
        self._rho_points = None

        self._diffc_perp = None
        self._dt = None
        self._n_t = None
        self._n_out = None

        self._rb = None
        self._rhob = None
        self._wrb = None
        self._wrhob = None

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
        self._coupling_mesh_name = data["interface"]["coupling_mesh_name"]
        try:
            self._write_data_name = data["interface"]["write_data_name"]
        except:
            self._write_data_name = None
        self._read_data_name = data["interface"]["read_data_name"]

        self._dims = data["mesh_parameters"]["dimensions"]
        self._rhomin = data["mesh_parameters"]["rho_inner"]
        self._rhomax = data["mesh_parameters"]["rho_outer"]
        self._r_spacing = data["mesh_parameters"]["radial_spacing"]
        self._rho_points = data["mesh_parameters"]["circular_points"]

        self._diffc_perp = data["diffusion_parameters"]["coeff_perp"]
        self._dt = data["simulation_parameters"]["timestep"]
        self._n_t = data["simulation_parameters"]["n_timesteps"]
        self._n_out = data["simulation_parameters"]["n_output"]

        self._rb = data["init_conditions"]["rb"]
        self._rhob = data["init_conditions"]["rho_point"]
        self._wrb = data["init_conditions"]["wrb"]
        self._wrhob = data["init_conditions"]["wrho_points"]

        read_file.close()

    def get_config_file_name(self):
        return self._config_file_name

    def get_participant_name(self):
        return self._participant_name

    def get_coupling_mesh_name(self):
        return self._coupling_mesh_name

    def get_read_data_name(self):
        return self._read_data_name

    def get_write_data_name(self):
        return self._write_data_name

    def get_dims(self):
        return self._dims

    def get_rhomin(self):
        return self._rhomin

    def get_rhomax(self):
        return self._rhomax

    def get_r_spacing(self):
        return self._r_spacing

    def get_rho_points(self):
        return self._rho_points

    def get_diffusion_coeff(self):
        return self._diffc_perp

    def get_dt(self):
        return self._dt

    def get_n_timesteps(self):
        return self._n_t

    def get_n_output(self):
        return self._n_out

    def get_rb(self):
        return self._rb

    def get_rhob(self):
        return self._rhob

    def get_wrb(self):
        return self._wrb

    def get_wrhob(self):
        return self._wrhob
