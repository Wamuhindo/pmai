import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Box, Discrete
from mushroom_rl.core.serialization import Serializable
from classes.Logger import Logger
from classes.System import System
from classes.Solution import Solution
from classes.ActionFactory import Actionf
from utils import get_current_timestamp, load_info_from_file, get_info_str, save_info_to_file, convert_index_to_state, \
    convert_state_to_index, define_decimals
import sys
import json
import math


class EnvironmentSEWQ(Environment, Serializable):
    
    """the class to represent the environment
    
        Attributes:
            _logger (Logger.Logger): the environment logger
            _current_step (int): the current step in the environment
            _step_per_checkpoint (int): the number of steps that the state of environment should be saved
            _error (Logger.Logger): the error logger
            _system (System.system): the system
            _initial_solution ( Solution.solution): the initial solution
            _actions (dict): dictionary of actions
            _previous_action (Action.Action): the action  at the timestep t-1
            _state (): the current system state
            _true_action (Action.Action): the action taken by the agent
            _current_configuration (AIapp.Configuration): the current system configuration
            _latency (float): the overall system latency at the current time step
            _weights (dict): the system weights considered in the cost function
            _gamma (float): the discount factor
            _horizon (int): the MDP horizon
            _varying_parameters: variable which specifies how mny parameters are varying. 
                                1:5G 
                                2:WIFI & 5G 
                                3: WIFI & 5G & l_cloud

    """    

    def __init__(self, system=None, system_file=None,
                 initial_sol=None, initial_sol_file=None,
                 env_params=None, env_params_file=None,
                 logger=Logger(), variations=[]):
        """initialize theenvironment

        Args:
            system (json object, optional): json object with the system config. Defaults to None.
            system_file (str, optional): file with the system config. Defaults to None.
            initial_sol (json object, optional): json object with the initial solution. Defaults to None.
            initial_sol_file (file, optional): file with the initial solution. Defaults to None.
            env_params (json object, optional): json object with environment parameters. Defaults to None.
            env_params_file (file, optional): file with environment parameters. Defaults to None.
            logger (logger, optional): the environment logger. Defaults to Logger().
            
        """        
        # Set the logger
        self._logger = logger
        self._current_step = 0
        self._error = Logger(stream=sys.stderr, verbose=1, is_error=True)
        self._discretization_value = 10
        self._energy = 0
        self._energy_Phone = 0
        
        # check if the mandatory file are provided
        self._check_parameters(system, system_file,
                               env_params,
                               env_params_file)
        self._logger.level += 1

        # initialize environment parameters
        self._logger.log("Initializing environment parameters...", 2)
        self._logger.level += 1
        params = {}
        setup_params = {}
        if env_params is not None:
            params = env_params["EnvParameters"]
        else:
            with open(env_params_file) as f:
                setup_params = json.load(f)
                params = setup_params["EnvParameters"]

        self._varying_parameters = variations

        
        if "Setup" not in setup_params.keys():
            self._error.log("Setup parameters not provided in the env_config file")
            sys.exit(1)
        else:
            if "n_steps" not in setup_params["Setup"].keys():
                self._error.log("the number of total steps n_steps not provided in the Setup parameters dictionary in the env_config file")
                sys.exit(1)
        self._max_env_step = setup_params["Setup"]["n_steps"]
        # initialize system
        self._logger.log("Initializing the system...", 2)
        if system is not None:
            self._system = system
        else:
            self._system = System(config_file=system_file,
                            logger=Logger(stream=self._logger.stream,
                                        verbose=self._logger.verbose,
                                        level=self._logger.level + 1), variations=self._varying_parameters,max_env_step=self._max_env_step)
            
        self._tmax = self._system.constraints.l_constraints["T_max"]
        # define initial configuration
        self._logger.log("Setting the initial solution", 2)
        if initial_sol is not None:
            self._initial_solution = initial_sol
        elif initial_sol_file is not None:
            self._initial_solution = Solution(self._system,
                                              initial_sol_file=initial_sol_file,
                                              logger=Logger(stream=self._logger.stream,
                                                            verbose=self._logger.verbose, level=self._logger.level+1))
        else:
            self._logger.log("No initial solution available", 2)
            self._initial_solution = Solution(self._system,
                                              logger=Logger(stream=self._logger.stream,
                                                            verbose=self._logger.verbose,
                                                            level=self._logger.level+1))
        self._current_configuration = self._initial_solution.configuration
        self.selected_action = 1



        self._initialize_parameters_env(params)
        self._logger.level -= 1
        self._logger.log("Environment parameters initialized successfully", 2)

        self._initial_solution = None

        # Create the action space.
        # setting the number of action to the number of configuration + 1(do_nothing)
        num_actions = len(self._system.configurations)

        self._actions = {}
        self._logger.log("Setting the set of possible actions...", 2)
        self._actions["0"] = Actionf.initialize("DoNothing", action_id=0)
        self._previous_action = self._actions["0"]
        self._true_action=None #just set an action that doesnt exists to initialize the variable
        for action_id in range(1, len(self._system.configurations)+1):
            self._actions[str(action_id)] = Actionf.initialize(
                "ChangeConfiguration", action_id=action_id, config=self._system.configurations[action_id-1])
        self._logger.log("Action set defined successfully", 2)

        action_space = Discrete(num_actions)

        # Create the observation space. It's a 2D box of dimension.
        # observation_space = Box(low=np.array(self._elements_min),
        #                        high=np.array(self._elements_max))

        observation_space = Discrete(
            num_actions * int(math.pow(self._discretization_value + 1, len(variations))))  # 23958   6181806 7986
        self.normalize_state = self.normalize_state_with_all_configs
        self.convert_to_original_state = self.convert_to_original_state_with_configs
        # Create the MDPInfo structure, needed by the environment interface
        mdp_info = MDPInfo(observation_space, action_space,
                           gamma=self._gamma, horizon=self._horizon)

        Environment.__init__(self, mdp_info)
        self._logger.log("MDP initialized successfully", 2)
        # Create a state class variable to store the current state
        self._state = None

        self._add_save_attr(
            _mdp_info='mushroom',
            _state='primitive',
            _current_step='primitive',
            _logger='pickle',
            _error='pickle',
            _varying_parameters='primitive',
            _tmax='primitive',
            _current_configuration='primitive',
            _actions='pickle',
            _previous_action='pickle',
            _true_action='pickle',
            _energy='primitive',
            _latency='primitive',
            _weights='primitive',
            _previous_cost='primitive',
            _reconfiguration_interval='primitive',
            _battery_variation_type='primitive',
            _battery_min_percentage='primitive',
            _battery_max_percentage='primitive',
            _battery_current_percentage='primitive',
            _variation_battery_weights='primitive',
            _horizon='primitive',
            _gamma='primitive',
            _initial_state='primitive',
            normalize_state='primitive',
            convert_to_original_state='primitive',
            _system='pickle',
            _initial_solution='pickle',
            _num_state_elts='primitive',
            _elements='primitive',
            _elements_min='primitive',
            _elements_max='primitive',
            _discretization_value='primitive',
            selected_action='primitive'

        )

    def reset(self, state=None, action = None):
        """reset the environment

        Args:
            state (_type_, optional): the state to wich to reset the environement. Defaults to None.

        Returns:
            state: the state to which the enviroment is reset
        """
        self._current_step = 0
        if action is None:
            self._true_action = self._actions[str(0)]
        else:
            self._true_action = action
        if state is None:
            # Generate randomly a state inside the state space, but not inside the goal
            self._state = self._initial_state
        else:
            # If an initial state is provided, set it and return, after checking it's valid.
            self._state = state

        # Return the current state
        return np.array([convert_state_to_index(self.normalize_state(self._state), self._discretization_value+1)])

    def step(self, _action):
        """take the action and make a step

        Args:
            _action : the acction that is taken by the agent

        Returns:
            np.array([state]), reward, absorbing, {}: return the state, the reward, the boolean to state if the current state is absorbing and a dictionnary to hold additionnal info
        """        
        # take the action
        # set the previous action on the environement the action taken
        # and do nothing if the action is the same as the on previously take
        action = _action[0]+1
        self.selected_action = int(action)
        self._current_step += 1
        if action not in range(0, len(self._actions.keys())):
            self._error.log("The chosen action is not in the action set")
            sys.exit(1)
        if self._true_action is not None:
            if int(action) != int(self._true_action.id):
                self._current_configuration = self._actions[str(
                    action)].apply(self._system.configurations)
                self._previous_action = self._actions[str(
                    action)]
                self._true_action = self._actions[str(action)]
            else:
                self._previous_action = self._actions["0"]
        else:
            self._current_configuration = self._actions[str(
                    action)].apply(self._system.configurations)
            self._previous_action = self._actions[str(
                    action)]
            self._true_action = self._actions[str(action)]
            

        self._logger.log("Taking action "+str(self._previous_action), 2)

        current_rate_WIFI = self._system.network_domains["WIFI7"]._get_rate(
            self._current_step)
        current_rate_5G = self._system.network_domains["5G"]._get_rate(
            self._current_step)
        battery = self._system.devices[0]._get_battery(self._current_step)[
            "value"]
        l_SEW = self._current_configuration.partitions[0].latency
        l_phone = self._current_configuration.partitions[1].latency
        l_cloud = self._system.devices[2]._get_latency(self._current_step,self._current_configuration)
        self._state = np.array(
            [current_rate_WIFI, current_rate_5G, battery, l_SEW, l_phone, l_cloud])
        print("state "+str(self._state))
        # Clip the state space inside the boundaries.
        # low = self.info.observation_space.low
        # high = self.info.observation_space.high
        # self._state = Environment._bound(self._state, low, high)
        # Compute the cost
        cost = self.compute_cost(
            self._system, self._state, self._current_configuration, self._previous_action)

        reward = -cost
        # Set the absorbing flag if SEW battery < min
        absorbing = False

        # return the index as state
        normalized_state = self.normalize_state(self._state)
        print("normalized state : " + str(normalized_state))
        state = convert_state_to_index(normalized_state, self._discretization_value+1)
        # Return all the information + empty dictionary (used to pass additional information)
        print("state_env "+str(state))
        return np.array([state]), reward, absorbing, {}

    def compute_cost(self, system, state, configuration, action):
        """compute the system cost

        Args:
            system (System.System): the system
            state (): the current state after the action
            configuration (AIapp.Configuration): the current configuration after the action
            action (Action.Action): the action taken

        Returns:
            _type_: _description_
        """        
        cost = 0
        cost_exec = 0
        cost_config = 0

        self._logger.level += 1
        self._logger.log("Computing costs...", 3)
        self._logger.level += 1
        # setting the reconfiguration cost
        if int(action.id) != 0:
            cost_config = 1.0
        self._logger.log("reconfiguration cost = {}".format(cost_config), 3)
        # compute the data transfer cost
        input_rate = system._input_rate
        data_transfer_cost = system.network_domains["5G"].cost_per_byte * \
             configuration.data_to_cloud
        if system._data_transfer_max_cost > 0:
            data_transfer_cost_normalized = data_transfer_cost / \
                system._data_transfer_max_cost
        else:
            data_transfer_cost_normalized = 0

        power_compute_phone = system.devices[1].power_compute
        power_compute_SEW = system.devices[0].power_compute
        power_transfer_phone = system.devices[1].power_transmission
        power_transfer_SEW = system.devices[0].power_transmission
        # computing energy SEW : power_compute is in mJ/MFLOP, and workload in MFLOP, power_transmission in Watt
        energy_cost_SEW = configuration.partitions[0].workload * \
            power_compute_SEW
        energy_compute_SEW = energy_cost_SEW
        data_rate_WIFI = state[0]/8
        data_rate_5G = state[1]/8
        if data_rate_WIFI == 0:
            data_rate_WIFI = 0.1
        if data_rate_5G == 0:
            data_rate_5G = 0.1
        # transfer energy SEW-PHONE
        transfer_time = 0
        if configuration.data_to_phone != 0:
            transfer_time = configuration.data_to_phone/data_rate_WIFI
        transfer_time = min(transfer_time, self._reconfiguration_interval)
        energy_cost_SEW += power_transfer_SEW * \
                           transfer_time
        # Computing energy PHONE
        energy_cost_phone = configuration.partitions[1].workload * \
            power_compute_phone
        energy_compute_phone = energy_cost_phone
        # transfer energy PHONE-CLOUD
        transfer_time = 0
        if configuration.data_to_cloud != 0:
            transfer_time = configuration.data_to_cloud/data_rate_5G
        transfer_time = min(transfer_time,self._reconfiguration_interval)
        energy_cost_phone += power_transfer_phone * \
            transfer_time
        energy_cost_normalized_SEW = energy_cost_SEW / (system._max_energy_SEW)
        energy_cost_normalized_Phone = energy_cost_phone / (system._max_energy_Phone)
        # if the computed cost is greater than the maximum one, this means
        # the data rate is 0 or too small, in that case if we would like to send the data
        # we would take more time, which implies the energy consumption would be greater
        # so we set the energy to the maximum to penalize the agent it it chooses to send data if in such a situation
        if energy_cost_normalized_SEW > 1:
            energy_cost_normalized_SEW = 1
        self._logger.log("energy cost SEW = {} (computing_energy:{}, data transfer energy:{})".format(
            energy_cost_normalized_SEW, energy_compute_SEW, energy_cost_SEW - energy_compute_SEW), 3)
        self._logger.log("energy cost SEW = {} (computing_energy:{}, data transfer energy:{})".format(
            energy_cost_normalized_Phone, energy_compute_phone, energy_cost_phone - energy_compute_phone), 3)
        self._energy = energy_cost_normalized_SEW
        self._energy_Phone = energy_cost_normalized_Phone
        # compute the overall latency
        _lcloud = 0
        if configuration.data_to_cloud != 0:
            _lcloud = state[5]
        latency = state[3]+state[4]+_lcloud + \
            (configuration.data_to_phone*1000 /
             data_rate_WIFI)+(configuration.data_to_cloud*1000/data_rate_5G)
        # give the penalty if execution time violation
        self._latency=latency
        lat_rate = 1
        #self._tmax=system.constraints.l_constraints["T_max"]
        if latency > self._tmax:
            cost_exec = 1.0
            lat_rate = min(1,1 )#(latency - self._tmax) / (self._tmax * 0.25)
        self._logger.log("Execution time violation cost = {}. (latency={})".format(
            cost_exec, latency), 3)
        # overall cost
        cost = self._weights["w_exec"]*cost_exec*lat_rate+self._weights["w_config"]*cost_config + \
            self._weights["w_conn"]*data_transfer_cost_normalized + \
            self._weights["w_energy_SEW"] * energy_cost_normalized_SEW + \
            self._weights["w_energy_Phone"] * energy_cost_normalized_Phone
        self._previous_cost = cost

        self._logger.log("Total system cost --> {}".format(cost), 3)
        # restore indentation level for logging
        self._logger.level -= 2

        return cost

    def render(self):
        """function to render the environment
        """ 
               
        return

    # get the enviroment infos
    def _get_info(self):
        """function to the get the info from the environement for logging

        Returns:
            dict: dictionnary that contains info
        """        

        info = dict()
        info["timestamp"] = get_current_timestamp()
        info["step"] = self._current_step
        info["state"] = self._state
        info["action"] = self._previous_action
        info["cost"] = self._previous_cost
        info["configuration"] = self._current_configuration
        info["battery"] = self._system.devices[0]._get_battery(
            self._current_step)
        info["l_cloud"] = self._system.devices[2]._get_latency(
            self._current_step,self._current_configuration)
        info["workload"] = self._system.devices[2]._get_workload(
            self._current_step)
        info["wifi"] = self._system.network_domains["WIFI7"]._get_rate(
            self._current_step)
        info["5G"] = self._system.network_domains["5G"]._get_rate(
            self._current_step)

        return info

    # get the info of the environnement as a string
    def _get_info_str(self, info):
        """convert info dictionnary to string

        Args:
            info (dict): info dictionnary

        Returns:
            _str: the info string
        """        
        return get_info_str(self._elements, info)

    # save the info of the environnement in a file
    def _save_info_to_file(self, info, new_info, file):
        """save the info into string

        Args:
            info (str): info string
            new_info (string): other information
            file (str): the file where to save the info
        """        
        info["others"] = new_info
        # print(new_info["Q"])
        save_info_to_file(self._get_info_str(info), file)

    # check if the system and the environment configuration file are provided
    def _check_parameters(self, system, system_file, env_params, env_params_file):
        """check the parameters of the system

        Args:
            system (json object): json object with system config
            system_file (str): file with system config
            env_params (json object): json object with environment parameters
            env_params_file (str): the file with the environemnt parameters 
        """
        if system is None and system_file is None:
            self._error.log("The system or the system file is not provided")
            sys.exit(1)

        if env_params is None and env_params_file is None:
            self._error.log(
                "No environment parameters or parameters file provided")
            sys.exit(1)

    # initialize the parameters of the environment
    def _initialize_parameters_env(self, params):
        """method to initialize the environment parameters

        Args:
            params (json object): json object with environment parameters
        """
        self._weights = {}

        if "ImportanceWeights" in params.keys():
            weights = params["ImportanceWeights"]
            for key in weights:
                self._weights[key] = weights[key]
        else:
            self._error.log(
                "No importance weights specified for the objective function")
            sys.exit(1)

        if "StateDiscretization" in params.keys():
            self._discretization_value = params["StateDiscretization"]
        else:
            self._error.log(
                "No discretization provided using (10) as default")
        if "StateParams" in params.keys():
            state_params = params["StateParams"]
            if "size" in state_params.keys():
                self._size = state_params["size"]
            else:
                self._logger.log(
                    "No state space size specified; setting an infinite MDP", 3)
                self._size = math.inf
            if "elements" in state_params.keys():
                self._elements = []
                self._elements_min = []
                self._elements_max = []
                self._initial_state = []
                elements = state_params["elements"]
                # setting the number of elements in the state tuple
                self._num_state_elts = len(elements.keys())
                self._logger.level += 1
                for el_name in elements.keys():
                    self._elements.append(el_name)
                    elt = elements[el_name]
                    if "min" in elt.keys():
                        self._elements_min.append(elt["min"])
                    else:
                        self._logger.log(
                            'Min value for state elt {} not provided; using default (0)'.format(el_name))
                        self._elements_min.append(0)
                    if "max" in elt.keys():
                        self._elements_max.append(elt["max"])
                    else:
                        self._logger.log(
                            'Max value for state elt {} not provided; using default (inf)'.format(el_name))
                        self._elements_max.append(math.inf)
                    if "initial" in elt.keys():
                        self._initial_state.append(elt["initial"])
                    else:
                        default_value = self.get_default_value(
                            el_name, self._initial_solution.configuration)
                        self._logger.log(
                            'Initial value for state elt {} not provided; using default ({})'.format(el_name, default_value))
                        self._initial_state.append(default_value)
                self._logger.level -= 1
            else:
                self._error.log(
                    "No state elements specified")
                sys.exit(1)

        else:
            self._error.log("State parameters not specified")
            sys.exit(1)

        if "LearningParams" in params.keys():
            learning_params = params["LearningParams"]
            if "gamma" in learning_params.keys():
                self._gamma = learning_params["gamma"]
            else:
                self._logger.log(
                    "Gamma not specified; using default (0.99)", 3)
                self._gamma = 0.99
        else:
            self._error.log("Learning parameters not specified")
            sys.exit(1)

        if "Horizon" in params.keys():
            # self._horizon = params["Horizon"]
            self._horizon = np.inf
        else:
            self._logger.log("No horizon specified; using default (100)", 3)
            self._horizon = np.inf
        if "ReconfigurationInterval" in params.keys():
            self._reconfiguration_interval = params["ReconfigurationInterval"]
        else:
            self._logger.log(
                "No reconfiguration interval specified; using default (5s)", 3)
            self._reconfiguration_interval = 5
        if "BatteryVariation" in params.keys():
            self._battery_variation_type = params["BatteryVariation"]["Type"]
            self._battery_min_percentage = params["BatteryVariation"]["MinPercentage"]
            self._battery_max_percentage = params["BatteryVariation"]["MaxPercentage"]
            self._battery_current_percentage = params["BatteryVariation"]["CurrentPercentage"]
            self._variation_battery_weights = params["BatteryVariation"]["WeightVariation"]
            # self._generate_battery_variations()
        else:
            self._logger.log("No battery variation function specified", 3)
            self._workload_variations = []
            self._next_bat_variation_idx = 0


    # set the verbosity level of the environment
    def set_verbosity(self, verbose):
        """set environment verbosity level

        Args:
            verbose (int): verbosity level
        """
        self._logger.verbose = verbose
        self._system.logger.verbose = verbose

    # get the initial state
    def get_default_value(self, elt, init_config):
        """ge the default vlue of the device parameters

        Args:
            elt (str): string that defines the element for which to get the initial value
            init_config (Solution.Solution): the initial configuration

        Returns:
            _type_: _description_
        """        
        if elt == "battery":
            return 100
        elif elt == "l_SEW":
            return init_config.partitions[0].latency
        elif elt == "l_phone":
            return init_config.partitions[1].latency
        else:
            return 0
    # load the environement informations from a file

    def _load_info_from_file(self, file):
        """load environment info from file

        Args:
            file str): the file
            
        Returns:
            dict: dictionnary with enironment info
        """        
        return load_info_from_file(file)

    def convert_to_original_state_with_configs(self, state):
        """convert the normalized state to the original state considering three source of variability

        Args:
            state (list): the state

        Returns:
            list: the original state
        """
        r_wifi_norm = None
        r_5G_norm = None
        l_cloud_norm = None
        # Here we get the normalized value by index. The depending on how they were put in the array while being normalized
        if "wifi" in self._varying_parameters:
            if "5G" in self._varying_parameters:
                if "cloud" in self._varying_parameters:
                    r_5G_norm = state[0]
                    r_wifi_norm = state[1]
                    l_cloud_norm = state[2]
                else:
                    r_5G_norm = state[0]
                    r_wifi_norm = state[1]
            else:
                if "cloud" in self._varying_parameters:
                    r_wifi_norm = state[1]
                    l_cloud_norm = state[0]
                else:
                    r_wifi_norm = state[0]
        else:
            if "5G" in self._varying_parameters:
                if "cloud" in self._varying_parameters:
                    r_5G_norm = state[0]
                    l_cloud_norm = state[1]
                else:
                    r_5G_norm = state[0]
            else:
                if "cloud" in self._varying_parameters:
                    l_cloud_norm = state[0]

        # The WIFI data rate is computed from state[4] that is the normalized value of the WIFI
        # here the original value can be computed as r_wifi = normalized_value *(max-min) + min
        # since min = 0, as r_wifi = normalized_value * max but to generalize it is better to keep it more general
        # The same applies to l_cloud and r_5G
        # Here value are None if they are not varying, then the maximum is taken for  WIFI7 and 5G and for l_cloud\
        #  the latency to execute partition3 on cloud
        battery = self._elements_max[2]
        config = self._system.configurations[self.selected_action - 1]
        if r_wifi_norm is not None:
            r_wifi = r_wifi_norm * (self._elements_max[0] - self._elements_min[0]) + self._elements_min[0]
        else:
            r_wifi = self._system.network_domains["WIFI7"].data_rate
        if l_cloud_norm is not None:
            l_cloud = l_cloud_norm * (self._elements_max[5] - self._elements_min[5]) + self._elements_min[5]
        else:
            l_cloud = config.partitions[3].latency
        if r_5G_norm is not None:
            r_5G = r_5G_norm * (self._elements_max[1] - self._elements_min[1]) + self._elements_min[1]
        else:
            r_5G = self._system.network_domains["5G"].data_rate

        # To get the l_SEW and l_Phone, we retrieve it from the partion choosen by taking an action.
        # In this way we generalize. Be don't need to write down all the 105 values from the action space
        l_SEW = config.partitions[0].latency
        l_phone = config.partitions[1].latency

        return [r_wifi, r_5G, battery, l_SEW, l_phone, l_cloud]

    def normalize_state_with_all_configs(self, state):
        """normalize the state considering three sources of variability discretized space

        Args:
            state (list): the state

        Returns:
            list: the normalized state
        """
        action = self.selected_action - 1

        # This is the generalized way to get the discretization of the state considering the execution times l_SEW and l_cloud
        # which depend on the chosen action. Note that this is valable for n_action < 11*11=121 for dicretization=10, if actions are more than 121,
        # we must add another state element s2 and s2=round(action//(11*11)/10.,1). We have the division by the dicretization_value to have the value in [0,1]
        s0 = round((action % (self._discretization_value + 1)) / self._discretization_value,
                   define_decimals(self._discretization_value))
        s1 = round((action // (self._discretization_value + 1)) / self._discretization_value,
                   define_decimals(self._discretization_value))
        # for each state element compute value_norm = (original_value - min )/ (max - min)
        # if min = 0; value_norm = original_value / max
        # then it value is rounded to x decimals to have  discrete values value per state feature, x is determined form the define_decimals function
        if "5G" in self._varying_parameters:
            r_5G_norm = round((state[1] - self._elements_min[1]) / (self._elements_max[1] - self._elements_min[1]),
                          define_decimals(self._discretization_value))
        if "wifi" in self._varying_parameters:
            r_wifi_norm = round((state[0] - self._elements_min[0]) / (self._elements_max[0] - self._elements_min[0]),
                            define_decimals(self._discretization_value))
        if "cloud" in self._varying_parameters:
            l_cloud_norm = round((state[5] - self._elements_min[5]) / (self._elements_max[5] - self._elements_min[5]),
                             define_decimals(self._discretization_value))

        if "wifi" in self._varying_parameters:
            if "5G" in self._varying_parameters:
                if "cloud" in self._varying_parameters:
                    return [0, s1, s0, l_cloud_norm, r_wifi_norm, r_5G_norm]

                else:
                    return [0, 0, s1, s0, r_wifi_norm, r_5G_norm]
            else:
                if "cloud" in self._varying_parameters:
                    return [0, 0, s1, s0, r_wifi_norm, l_cloud_norm]
                else:
                    return [0, 0, 0, s1, s0, r_wifi_norm]
        else:
            if "5G" in self._varying_parameters:
                if "cloud" in self._varying_parameters:
                    return [0, 0, s1, s0, l_cloud_norm, r_5G_norm]
                else:
                    return [0, 0, 0, s1, s0, r_5G_norm]
            else:
                if "cloud" in self._varying_parameters:
                    return [0, 0, 0, s1, s0, l_cloud_norm]
                else:
                    return [0, 0, 0, 0, s1, s0]


EnvironmentSEWQ.register()
