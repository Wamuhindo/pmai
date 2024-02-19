

from classes.Logger import Logger
from classes.AIapp import Configuration, Partition
from classes.Constraint import Constraint
from classes.NetworkDomain import NetworkDomain
from classes.Device import Device
from classes.DeviceFactory import Devicef
from utils import generate_keywords_dict

import numpy as np
import json
import sys


class System:
    """ the system class
    
        Args:
            config_file (str): the system configuration file
            json_config (json): the system json object
            logger (Logger.Logger) : the  logger object 
        
        Attributes:
            logger (Logger.Logger): the system logger object 
            error (Logger.Logger): the logger for the error 
            _input_rate (float) : the system input rate
            _max_energy_transfer_SEW (float): the maximum energy for tansfering the tensor to the phone (needed to normalize the energy)     
            _max_energy_transfer_phone (float): the maximum energy for tansfering the tensor to the cloud (needed to normalize the energy) 
            PHONE (Device.Device): the phoen Device
            SEW (Device.Device): the SEW device
            CLOUD (Device.Device): the cloud Device
            devices (list(Device.Devices): list of system devices     
            constraints (list(Constraint.Constraint)) :list of system contraints  
            network_domains (list(NetworkDomain.NetworkDomain))
            _max_data_to_cloud (float): the maximum amount of data that can be transfer to the cloud(used for normalization)
            _max_data_to_phone (float): the maximum amout of data that can be transfered to the phone (used fo normalization)
            _max_partition_workload (float): the maximum partiton workload of the system
            configurations (list(AIapp.Configuration)) : the list of all system configurations
        
     """
    def __init__(self, config_file=None, json_config=None, logger=Logger(),variations=[],env_number=1,max_env_step=10000):
       
        self.logger = logger
        self.error = Logger(stream=sys.stderr, verbose=1, is_error=True)
        self._varying_parameters = variations
        self.env_number = env_number
        self.max_env_step = max_env_step
        if config_file:
            self.logger.log("Loading the system configuration from file...", 1)
            self.load_config_file(config_file)
        elif json_config:
            self.logger.log(
                "Loading system configuration from json object...", 1)
            self.load_json_config(json_config)
        else:
            self.error.log("No configuration file or json specified")
            self.error.log("System exiting...")
            sys.exit(1)

    def load_config_file(self, file):
        """method to load the system configuration file

        Args:
            file (str): config file
        """
        with open(file) as f:
            data = json.load(f)
        self.load_json_config(data)

    def load_json_config2(self, json_config):
        """method to load the system from the json object

        Args:
            json_config (Json Object): json object that contains the system configuration
        """        
        print(json_config)

    def load_json_config(self, json_config):
        """method to load the system from the json object

        Args:
            json_config (Json Object): json object that contains the system configuration
        """        

        self.logger.log("Initialisation of the System...", 1)
        # increase indentation level for logging
        self.logger.level += 1

        # generate dictionaries of keywords
        keywords = generate_keywords_dict(json_config.keys())

        # initialize components
        if "Configurations" in keywords.keys():
            self.logger.log("Initializing configurations...", 1)
            self.initialize_configurations(
                json_config[keywords["Configurations"]])
        else:
            self.error.log("No configuration available in configuration file")
            sys.exit(1)
        self._input_rate = 0
        if "InputRate" in keywords.keys():
            self._input_rate = json_config[keywords["InputRate"]]
        else:
            self.error.log(
                "System input rate not defined")
            sys.exit(1)
        # get constraints
        constraints = None
        if "Constraints" in keywords.keys():
            self.logger.log("Initializing system constraints...", 1)
            constraints = json_config[keywords["Constraints"]]
        else:
            self.error.log("System constraints not provided")
            sys.exit(1)
        self.initialize_constraints(constraints)
        #
        if "ReconfigurationInterval" in keywords.keys():
            self._reconfiguration_interval = json_config[keywords["ReconfigurationInterval"]]
        else:
            self.logger.log(
                "reconfiguration interval not provided using default (10) ")
            self._reconfiguration_interval = 10
        '''if "EnergyTransfer" in keywords.keys():
            energyTransfer = json_config[keywords["EnergyTransfer"]]
            if "max_energy_transfer_SEW" in energyTransfer.keys():
                self._max_energy_transfer_SEW = energyTransfer["max_energy_transfer_SEW"]
            else:
                self.error.log("Max energy tranfer for the SEW not specified")
                sys.exit(1)
            if "max_energy_transfer_phone" in energyTransfer.keys():
                self._max_energy_transfer_phone = energyTransfer["max_energy_transfer_phone"]
            else:
                self.error.log(
                    "Max energy tranfer for the phone not specified")
                sys.exit(1)
        else:
            self.error.log(
                "Energy trnasfer (max) for SEW and Phone not specified")
            sys.exit(1)'''

        # initialize devices
        self.logger.log("Initializing devices...", 1)
        self.initialize_devices(json_config, keywords)

        if "NetworkDomains" not in keywords.keys():
            self.error.log("Network domains not provided")
            sys.exit(1)
        else:
            self.logger.log("Initializing network domains...", 1)
            self.initialize_network_domains(
                json_config[keywords["NetworkDomains"]])
        self.logger.level -= 1
        self.logger.log("System initialized sucessfully", 1)

        self.initialize_max_costs()

    def initialize_devices(self, data, keywords):
        """
        Initialize the list of devices

        """
        self.PHONE = None
        self.SEW = None
        self.CLOUD = None
        self.devices = []

        # increase indentation level for logging
        self.logger.level += 1

        if "Devices" in keywords.keys():
            _keywords = data["Devices"]
            # SEW device
            if "SEW" in _keywords.keys():
                self.logger.log("Smart Eye-Wear device...", 1)
                SEW = data["Devices"]["SEW"]
                self.SEW = self.add_device("SEW", SEW, 1)
            else:
                self.error.log("SEW not available")
                sys.exit(1)

            # Edge resource : mobile phone
            if "Phone" in _keywords.keys():
                self.logger.log("Smart phone...", 1)
                PHONE = data["Devices"]["Phone"]
                self.PHONE = self.add_device("Phone", PHONE, 2)
            else:
                self.error.log("Mobile phone not available")
                sys.exit(1)
            # Cloud resources
            if "Cloud" in _keywords.keys():
                self.logger.log("Cloud server...", 1)
                CLOUD = data["Devices"]["Cloud"]
                self.CLOUD = self.add_device("Cloud", CLOUD, 3)
            else:
                self.error.log("Cloud server not available")
                sys.exit(1)
            # restore indentation level for logging
            self.logger.level -= 1
            self.logger.log("All devices added successfully", 1)
        else:
            self.error.log("Devices are not availables")
            sys.exit(1)

    def add_device(self, device_type, device_dictionnary, position):
        """method that helps add a device

        Args:
            device_type (str): the device type
            device_dictionnary (dict): the dictionnary of all devices
            position (int): position where to add the device

        Returns:
            _type_: _description_
        """
        new_device = Devicef.initialize(
            device_type, name=device_type, variations=self._varying_parameters, **device_dictionnary)
        self.devices.insert(position, new_device)
        self.logger.log(
            'The Device "{}" added successfully'.format(device_type), 1)

        return new_device

    def initialize_constraints(self, constraints):
        """methods to initialize the list of constraints

        Args:
            constraints (list): list of constraints
        """
        self.constraints = Constraint(constraints)
        self.logger.log("System constraints set successfully", 1)

    def initialize_network_domains(self, net_domains):
        """method to initialize the network domain

        Args:
            net_domains (dict): dictionary of network domains
        """        
        self.network_domains = {}

        for net_name in net_domains.keys():
            net = net_domains[net_name]
            self.network_domains[net_name] = NetworkDomain(
                name=net_name,variations=self._varying_parameters,env_number=self.env_number,max_env_step=self.max_env_step, **net)

        self.logger.log("Network domains set succesfully", 1)

    def initialize_max_costs(self):
        """methods to initilaize the maximum costs of the system
        """        
        # getting the max data amount that can be transfered to the cloud
        self._max_data_to_cloud = 0
        self._max_data_to_phone = 0
        for configuration in self.configurations:
            if configuration.data_to_cloud > self._max_data_to_cloud:
                self._max_data_to_cloud = configuration.data_to_cloud
            if configuration.data_to_phone > self._max_data_to_phone:
                self._max_data_to_phone = configuration.data_to_phone
        # getting the max energy cost that can be incured
        self._data_transfer_max_cost = self.network_domains["5G"].cost_per_byte * \
            self._max_data_to_cloud
        # get the maximum workload among the configurations
        self._max_partition_workload_phone = 0
        self._max_partition_workload_SEW = 0
        for configuration in self.configurations:
            if configuration.partitions[0].workload > self._max_partition_workload_SEW:
                self._max_partition_workload_SEW = configuration.partitions[0].workload
            if configuration.partitions[1].workload > self._max_partition_workload_phone:
                self._max_partition_workload_phone = configuration.partitions[1].workload
        power_compute_phone = self.devices[1].power_compute
        power_compute_SEW = self.devices[0].power_compute
        power_transfer_phone = self.devices[1].power_transmission
        power_transfer_SEW = self.devices[0].power_transmission
        self._max_energy_compute_SEW = self._max_partition_workload_SEW * \
                                       power_compute_SEW
        self._max_energy_compute_Phone = self._max_partition_workload_phone * \
                                         power_compute_phone
        self._max_energy_SEW = self._max_energy_compute_SEW + \
                               power_transfer_SEW * self._reconfiguration_interval
        self._max_energy_Phone = self._max_energy_compute_Phone + \
                                 power_transfer_phone * self._reconfiguration_interval
        self.logger.log("max_energy SEW {}".format(
            self._max_energy_SEW), 3)
        self.logger.log("max_energy Phone {}".format(
            self._max_energy_Phone), 3)

    def initialize_configurations(self, configurations):
        """method to initialize the system configurations

        Args:
            configurations (Object): object of configurations
        """        
        self.configurations = []

        self.logger.level += 1
        for config_name in configurations:
            partitions = []
            for part_name in configurations[config_name]["partitions"]:
                part = configurations[config_name]["partitions"][part_name]
                if "workload" not in part.keys():
                    msg = 'workload property not provided for partition {} in configuration {}'.format(
                        part_name, config_name)
                    self.error.log(msg)
                    sys.exit(1)
                elif "latency" not in part.keys():
                    msg = 'latency property not provided for partition {} in configuration {}'.format(
                        part_name, config_name)
                    self.error.log(msg)
                    sys.exit(1)
                elif "name" not in part.keys():
                    msg = 'name property not provided for partition {} in configuration {}'.format(
                        part_name, config_name)
                    self.error.log(msg)
                    sys.exit(1)
                else:
                    partition = Partition(fake_name=part_name, **part)
                    partitions.append(partition)

            if "data_to_cloud" not in configurations[config_name].keys():
                msg = 'data_to_cloud property not provided for configuration {}'.format(
                    config_name)
                self.error.log(msg)
                sys.exit(1)
            elif "data_to_phone" not in configurations[config_name].keys():
                msg = 'data_to_phone property not provided for configuration {}'.format(
                    config_name)
                self.error.log(msg)
                sys.exit(1)
            else:
                d_phone = configurations[config_name]["data_to_phone"]
                d_cloud = configurations[config_name]["data_to_cloud"]
                self.configurations.append(Configuration(
                    name=config_name, data_to_cloud=d_cloud, data_to_phone=d_phone, partitions=partitions))
        self.logger.level -= 1
        self.logger.log("DNN Configurations set successfully", 1)

    def to_json(self):
        """the method to save the system into a json object

        Returns:
            json object: the json object created from the system
        """        

        # configurations
        system_string = '{"Configurations": {'
        for config in self.configurations:
            system_string += (str(config) + ',')
        system_string = system_string[:-1] + '}'

        # devices
        system_string += ', '

        devices_string = '\n"Devices": {'

        for device in self.devices:
            devices_string += (str(device) + ',')

        devices_string = devices_string[:-1] + '}'
        devices_string += ', '
        system_string += devices_string
        # Constraints
        system_string += self.constraints.__str__()

        # Network Domains
        system_string += ', \n"NetworkDomains": {'
        for net_domain in self.network_domains:
            system_string += (self.network_domains[net_domain].__str__() + ',')
        system_string = system_string[:-1] + '}'
        '''system_string += ',"EnergyTransfer":{'
        system_string += '"max_energy_transfer_SEW":{},"max_energy_transfer_phone":{}'.format(
            self._max_energy_transfer_SEW, self._max_energy_transfer_phone)
        system_string += '}' '''
        system_string += ', "ReconfigurationInterval":{}'.format(self._reconfiguration_interval)
        system_string += ', "InputRate":{}'.format(self._input_rate)
        system_string += '}'

        # load the system string as json
        system_config = json.dumps(json.loads(system_string), indent=2)

        return system_config

    def print_system(self, file=""):
        """method to print the systemeitherr in a file or on the standard output

        Args:
            file (str, optional): the file in which to save the system config. Defaults to "".
        """        

        # get system description in json format
        desc = self.to_json()

        # print

        if file:
            with open(file, "w") as f:
                f.write(desc)
        else:
            print(desc)
