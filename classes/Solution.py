from classes.Logger import Logger
from classes.System import System
from classes.AIapp import Configuration, Partition
from utils import generate_keywords_dict
import json
import sys
import random


class Solution:

    def __init__(self, system: System = None, initial_sol_json=None, initial_sol_file=None, logger=Logger) -> None:

        self.logger = logger
        self.error = Logger(stream=sys.stderr, verbose=1, is_error=True)

        if system is not None:
            self.logger.log("Initializing an empty solution", 1)
            self.initialize_solution(system)

            if initial_sol_file:
                self.logger.log("Loading solution configuration from file", 1)
                self.load_file(system, initial_sol_file)

            elif initial_sol_json:
                self.logger.log(
                    "Loading solution configuration from json object", 1)
                self.load_json(system, initial_sol_json)
            else:
                system_configs = system.configurations
                self.configuration = system_configs[random.randint(
                    0, len(system_configs)-1)]
        else:
            self.error.log("System is not passed to the solution")
            sys.exit(1)

    def initialize_solution(self, system: System):

        self.C = len(system.configurations)
        self.P = [len(configuration.partitions)
                  for configuration in system.configurations]
        self.D = len(system.devices)

    def load_file(self, system, initial_sol_file):

        # load json file
        with open(initial_sol_file) as f:
            data = json.load(f)

        self.load_json(system, data)

    def load_json(self, system, initial_sol_json):

        # increase indentation level for logging
        self.logger.level += 1

        # generate dictionaries of keywords
        keywords = generate_keywords_dict(initial_sol_json.keys())

        # configuration and  corresponding partitions
        if "Configuration" in keywords.keys():
            key = keywords["Configuration"]
            self.initialize_configuration(system, initial_sol_json[key])
            self.logger.level -= 1
            self.logger.log("DNN Solution Configuration set successfully", 1)
        else:
            self.error.log("No solution configuration given.")
            sys.exit(1)

        # total cost
        if "totalcost" in keywords.keys():
            self.total_cost = initial_sol_json[keywords["totalcost"]]
        else:
            self.error.log(
                "No total cost in solution configuration description")
            sys.exit(1)

        # restore indentation level for logging
        self.logger.level -= 1

    def initialize_configuration(self, system, configuration):

        self.configuration = None

        self.logger.level += 1
        for config_name in configuration:
            partitions = []
            for part_name in configuration[config_name]["partitions"]:
                part = configuration[config_name]["partitions"][part_name]
                if "workload" not in part.keys():
                    msg = 'workload property not provided for partition {} in solution configuration {}'.format(
                        part_name, config_name)
                    self.error.log(msg)
                    sys.exit(1)
                elif "latency" not in part.keys():
                    msg = 'latency property not provided for partition {} in  solution configuration {}'.format(
                        part_name, config_name)
                    self.error.log(msg)
                    sys.exit(1)
                elif "name" not in part.keys():
                    msg = 'name property not provided for partition {} in  solution configuration {}'.format(
                        part_name, config_name)
                    self.error.log(msg)
                    sys.exit(1)
                else:
                    partition = Partition(fake_name=part_name, **part)
                    partitions.append(partition)

            if "data_to_cloud" not in configuration[config_name].keys():
                msg = 'data_to_cloud property not provided for  solution configuration {}'.format(
                    config_name)
                self.error.log(msg)
                sys.exit(1)
            elif "data_to_phone" not in configuration[config_name].keys():
                msg = 'data_to_phone property not provided for  solution configuration {}'.format(
                    config_name)
                self.error.log(msg)
                sys.exit(1)
            else:
                d_phone = configuration[config_name]["data_to_phone"]
                d_cloud = configuration[config_name]["data_to_cloud"]
                self.configuration = Configuration(
                    name=config_name, data_to_cloud=d_cloud, data_to_phone=d_phone, partitions=partitions)
        self.logger.level -= 1

    def to_json(self):

        # configurations
        solution_string = '{"Configuration": {'
        solution_string += (str(self.configuration))
        solution_string += '}'

        # devices
        solution_string += ', '

        solution_string += '\n"total_cost":'+str(self.total_cost)

        solution_string += '}'

        # load the system string as json
        solution_config = json.dumps(json.loads(solution_string), indent=2)

        return solution_config

    def print_configuration(self, solution, sol_file=""):

        # get solution description in json format
        sol = self.to_json(solution)

        # print
        if sol_file:
            with open(sol_file, "w") as f:
                f.write(sol)
        else:
            print(sol)
