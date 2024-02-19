import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Box, Discrete
from mushroom_rl.core.serialization import Serializable
from classes.Logger import Logger
from classes.System import System
from classes.Solution import Solution
from classes.ActionFactory import Actionf
from utils import get_current_timestamp, load_info_from_file, get_info_str, save_info_to_file, \
    convert_index_to_state, convert_state_to_index, define_decimals
import sys
import json
import math
import random
import copy


class EnvSingle():
    def __init__(self,logger):
        self.logger = logger
        self._previous_action = None
        self._true_action = None
        self._current_configuration = None
        self._system = None
        self._weights = {}
        self._tmax = None
        self._reward = -1.5
        self._latency = None
        self._wifi=0
        self._5G =0
        self._l_cloud=0
        self._energy_SEW=0
        self._energy_Phone=0
        self._current_step = 0
    def __str__(self):
        """EnvSingle in String
        """
        s = f'''{{"previous_action":{{{self._previous_action}}},"true_action":{{{self._true_action}}}, "reward":{self._reward}, "latency":{self._latency},"tmax":{self._tmax},"wifi":{self._wifi},"5G":{self._5G},"l_cloud":{self._l_cloud},"energy_SEW":{self._energy_SEW},"energy_Phone":{self._energy_Phone}}}'''
        return s

class EnvironmentSEWDQNExtended(Environment, Serializable):
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
                 logger=Logger(), variations=[], weights=[],tmaxs=[],seed=1234,steps_per_env=5,all_steps_per_env=False,round_robin=True,evaluation=True):
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
        self._agent = None
        self._initial = True
        self._steps_per_env = steps_per_env
        self._logger = logger
        self._seed = seed
        self._current_step = 0
        self._weights = {}
        self._error = Logger(stream=logger.stream if logger.stream != sys.stdout else sys.stderr, verbose=1, is_error=True)
        self._discretization_value = 10
        self._energy = 0
        self._energy_Phone = 0
        self._all_steps_per_env = all_steps_per_env
        self.chosen_env = 0
        self._round_robin = round_robin
        self._evaluation = evaluation
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
        # initialize system
        self._logger.log("Initializing the system...", 2)

        self.selected_action = 1
        self.selected_env = 0
        self.steps_for_current_env = 0
        

        self._initialize_parameters_env(params)
        self._logger.level -= 1
        self._logger.log("Environment parameters initialized successfully", 2)
        _envs = []

        i=0
        if "Setup" not in setup_params.keys():
            self._error.log("Setup parameters not provided in the env_config file")
            sys.exit(1)
        else:
            if "n_steps" not in setup_params["Setup"].keys():
                self._error.log("the number of total steps n_steps not provided in the Setup parameters dictionary in the env_config file")
                sys.exit(1)
        self._max_env_step = setup_params["Setup"]["n_steps"]

        for weight in weights:
            for tmax in tmaxs:
                i+=1
                env_single = self._init_single_env(weight,tmax,i,initial_sol,initial_sol_file,system_file)
                _envs.append(env_single)
        self._envs = _envs
        np.random.seed(self._seed)
        random.seed(self._seed)

        # Create the action space.
        # setting the number of action to the number of configuration + 1(do_nothing)
        num_actions = len(self._envs[0]._system.configurations)

        self._actions = {}
        self._logger.log("Setting the set of possible actions...", 2)
        self._actions["0"] = Actionf.initialize("DoNothing", action_id=0)
        self._previous_action = self._actions["0"]
        self._true_action = None  # just set an action that doesnt exists to initialize the variable
        for action_id in range(1, len(self._envs[0]._system.configurations) + 1):
            self._actions[str(action_id)] = Actionf.initialize(
                "ChangeConfiguration", action_id=action_id, config=self._envs[0]._system.configurations[action_id - 1])
        self._logger.log("Action set defined successfully", 2)

        action_space = Discrete(num_actions)

        # Create the observation space. It's a 2D box of dimension. #self._elements_min  self._elements_max
        observation_space = Box(low=np.array([0]*len(self._elements_min)),
                                high=np.array([1]*len(self._elements_max)))

        self.normalize_state = self.normalize_state_with_all_configs
        self.convert_to_original_state = self.convert_to_original_state_with_configs
        # Create the MDPInfo structure, needed by the environment interface
        mdp_info = MDPInfo(observation_space, action_space,
                           gamma=self._gamma, horizon=self._horizon)
        

        Environment.__init__(self, mdp_info)

        # Create a state class variable to store the current state
        self._state = None
        self._envs_id = [item for item in range(len(self._envs))]


        print(self._max_env_step)

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
            _envs='pickle',
            _envs_id='primitive',
            _all_steps_per_env = 'primitive',
            chosen_env = 'primitive',
            _round_robin = 'primitive',
            _evaluation = 'primitive'

        )
 

    def reset(self, state=None, action = None):
        """reset the environment

        Args:
            state (_type_, optional): the state to wich to reset the environement. Defaults to None.

        Returns:
            state: the state to which the enviroment is reset
        """
        self._current_step = 0
        self._initial = True
        if action is None:
            self._true_action = self._actions[str(0)]
        else:
            self._true_action = action
        if state is None:
            # Generate randomly a state inside the state space, but not inside the goal
            self._state = self._envs[0]._initial_state
        else:
            # If an initial state is provided, set it and return, after checking it's valid.
            self._state = state

        # Return the current state
        return np.array(self.normalize_state(self._state)) #np.array([convert_state_to_index(self.normalize_state(self._state), 11)])

    def step(self, _action):
        """take the action and make a step

        Args:
            _action : the acction that is taken by the agent

        Returns:
            np.array([state]), reward, absorbing, {}: return the state, the reward, the boolean to state if the current state is absorbing and a dictionnary to hold additionnal info
        """
        self._logger.log(f"chosen_env : {self.chosen_env}",1)
        for i in range(len(self._envs)):
            self._logger.log(f"env{i} current step : {self._envs[i]._current_step}",1)
        if self._initial or len(self._envs_id)==0:
            self._envs_id = [item for item in range(len(self._envs))]

        #If the agent has run in the previous chosen environment for _steps_per_env steps, chose another environment
        if self.steps_for_current_env > self._steps_per_env or self._initial or self._envs[self.chosen_env]._current_step >= self._max_env_step:
            if self._initial:
                self.chosen_env = 0
            else:
                if self._round_robin:
                    self.chosen_env +=1
                    if self.chosen_env == len(self._envs):
                        self.chosen_env = 0
                else: 
                    self.chosen_env = random.choice(self._envs_id)
                #If the chosen env has reached the maximum step no need to choose it
                if self._envs[self.chosen_env]._current_step >= self._max_env_step:
                    if not self._round_robin:
                        self._envs_id.remove(self.chosen_env)
                        env_id = self.chosen_env
                        while self.chosen_env == env_id:
                            if len(self._envs_id)==0:
                                for i in range(len(self._envs)):
                                    if self._envs[i]._current_step < self._max_env_step:
                                        self._envs_id.append(i)
                            self.chosen_env = random.choice(self._envs_id)
                    else:
                        self.chosen_env +=1
                        if self.chosen_env == len(self._envs):
                            self.chosen_env = 0
                            while self._envs[self.chosen_env]._current_step >= self._max_env_step:
                                self.chosen_env += 1
                            

            self.steps_for_current_env = 0
            self._initial = False   
        
        
        if not self._round_robin and self.chosen_env in self._envs_id:
            #remove the chosen environment in the set of environment to be chosen
            self._envs_id.remove(self.chosen_env)     
        
        action_ = _action[0] + 1
        self.selected_action = int(action_)

        self._current_step += 1
        if action_ not in range(0, len(self._actions.keys())):
            self._error.log("The chosen action is not in the action set")
            sys.exit(1)
        #make a step in the chosen environment
        if self._envs[self.chosen_env]._current_step < self._max_env_step:
            env = self._step_in_env(self._envs[self.chosen_env],action_)
            self._envs[self.chosen_env] = env
        self._logger.log(f"non normalized stat chosen env {self._envs[self.chosen_env]._previous_state}",2)



        self._logger.log(f"Taking action {str(self._envs[self.chosen_env]._previous_action)} {str(action_)} in env {self.chosen_env} ", 1)

        if not self._all_steps_per_env:
            #retrieve the action for each environment, make a state in it and retrieve the state
            for i in range(len(self._envs)):
                if i != self.chosen_env:
                    env = self._envs[i]
                    state = env._previous_state if env._previous_state is not None else env._initial_state
                    #self._logger.log(f"previous stat {env._previous_state}",2)
                    state = self.normalize_state(state)
                    #self._logger.log(f"normalized state {state}",2)
                    _action = self._agent.draw_action(state)
                    action = _action[0] + 1
                    self._logger.log(f"action env{i}:  {action}",2)

                    # take the action
                    # set the previous action on the environement the action taken
                    # and do nothing if the action is the same as the on previously take
                    env = self._step_in_env(env,action)
                    self._logger.log(f"non-normalized state {env._previous_state}",2)
                    self._logger.log(f"normalized state {self.normalize_state(env._previous_state)}",2)

        

        self._state = np.array(
            self._envs[self.chosen_env]._previous_state)
        #[current_rate_WIFI,current_rate_5G,l_SEW,l_phone,l_cloud,w_exec,w_conn,w_config,w_energy_Phone,w_energy_SEW,threshold] = self._envs[self.chosen_env]._previous_state
        #print("state : " + str({"r_wifi":current_rate_WIFI,"r_5G":current_rate_5G,"l_SEW":l_SEW,"l_phone":l_phone,"l_cloud":l_cloud,"w_exec":w_exec,"w_con":w_conn,"w_config":w_config,"w_energy_phone":w_energy_Phone,"w_energy_SEW":w_energy_SEW,"tmax":threshold}))

        # Clip the state space inside the boundaries.
        # low = self.info.observation_space.low
        # high = self.info.observation_space.high

        # self._state = Environment._bound(self._state, low, high)

        # Compute the cost
        cost = self.compute_cost(
            self._envs[self.chosen_env]._system, self._state, self._envs[self.chosen_env]._current_configuration, 
            self._envs[self.chosen_env]._previous_action,self._envs[self.chosen_env])
        # Compute the reward
        # since the cost is always between 0 and 1 we can compute the reward as 1-cost and maximize the cumulative reward
        # this way the network will have the relu as activation function
        reward = 1-cost

        if not self._all_steps_per_env:
        #compute the reward in the others no choosen environmemnts
            for idx in range(len(self._envs)):
                if idx != self.chosen_env:
                    env = self._envs[idx]
                    self.compute_cost(env._system, env._previous_state, 
                                    env._current_configuration, 
                                    env._previous_action,env)



        # Set the absorbing flag if SEW battery < min
        absorbing = False

        # return the index as state
        state = self.normalize_state(self._state) #convert_state_to_index(self.normalize_state(self._state), 11)
        # Return all the information + empty dictionary (used to pass additional information)
        self._logger.log("state_env norm" + str(self._state),1)

        self.steps_for_current_env +=1
        #return np.array([state]), reward, absorbing, {}
        return np.array(state), reward, absorbing, {}

    def compute_cost(self, system, state, configuration, action, env):
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
        #self._logger.log("reconfiguration cost = {}".format(cost_config), 3)
        # compute the data transfer cost
        input_rate = system._input_rate
        data_transfer_cost = system.network_domains["5G"].cost_per_byte * \
                             configuration.data_to_cloud
        if system._data_transfer_max_cost > 0:
            data_transfer_cost_normalized = data_transfer_cost / system._data_transfer_max_cost
        else:
            data_transfer_cost_normalized = 0
        power_compute_phone = system.devices[1].power_compute
        power_compute_SEW = system.devices[0].power_compute
        power_transfer_phone = system.devices[1].power_transmission
        power_transfer_SEW = system.devices[0].power_transmission
        # computing energy SEW
        energy_cost_SEW = configuration.partitions[0].workload * \
                          power_compute_SEW
        energy_compute_SEW = energy_cost_SEW
        data_rate_WIFI = state[0] / 8
        data_rate_5G = state[1] / 8
        if data_rate_WIFI == 0:
            data_rate_WIFI = 0.1
        if data_rate_5G == 0:
            data_rate_5G = 0.1
        # transfer energy SEW-PHONE
        transfer_time = 0
        if configuration.data_to_phone != 0:
            transfer_time = configuration.data_to_phone / data_rate_WIFI
        transfer_time = min(transfer_time, self._reconfiguration_interval)
        energy_cost_SEW += power_transfer_SEW * transfer_time
        # computing energy SEW : power_compute is in mJ/MFLOP, and workload in MFLOP, power_transmission in Watt
        energy_cost_phone = configuration.partitions[1].workload * power_compute_phone
        energy_compute_phone = energy_cost_phone
        # transfer energy PHONE-CLOUD
        transfer_time = 0
        if configuration.data_to_cloud != 0:
            transfer_time = configuration.data_to_cloud / data_rate_5G
        transfer_time = min(transfer_time, self._reconfiguration_interval)
        energy_cost_phone += power_transfer_phone * transfer_time

        energy_cost_normalized_SEW = energy_cost_SEW / (system._max_energy_SEW)
        energy_cost_normalized_Phone = energy_cost_phone / (system._max_energy_Phone)
        # if the computed cost is greater than the maximum one, this means
        # the data rate is 0 or too small, in that case if we would like to send the data
        # we would take more time, which implies the energy consumption would be greater
        # so we set the energy to the maximum to penalize the agent it it chooses to send data if in such a situation
        if energy_cost_normalized_SEW > 1:
            energy_cost_normalized_SEW = 1
        '''self._logger.log("energy cost SEW = {} (computing_energy:{}, data transfer energy:{})".format(
            energy_cost_normalized_SEW, energy_compute_SEW, energy_cost_SEW - energy_compute_SEW), 3)
        self._logger.log("energy cost SEW = {} (computing_energy:{}, data transfer energy:{})".format(
            energy_cost_normalized_Phone, energy_compute_phone, energy_cost_phone - energy_compute_phone), 3)'''
        self._energy = energy_cost_normalized_SEW
        self._energy_Phone = energy_cost_normalized_Phone
        # compute the overall latency
        _lcloud = 0
        if configuration.data_to_cloud != 0:
            _lcloud = state[4]
        latency = state[2] + state[3] + _lcloud + \
                  (configuration.data_to_phone * 1000 /
                   data_rate_WIFI) + (configuration.data_to_cloud * 1000 / data_rate_5G)
        # give the penalty if execution time violation
        self._latency = latency
        env._latency = latency
        lat_rate = 1
        # self._tmax=system.constraints.l_constraints["T_max"]
        if latency > env._tmax:
            cost_exec = 1.0
            lat_rate = min(1, 1)#(latency - env._tmax) / (env._tmax * 0.15)
        self._logger.log("Execution time violation cost = {}. (latency={})".format(
            cost_exec, latency), 3)
        # overall cost
        cost = env._weights["w_exec"] * cost_exec*lat_rate + env._weights["w_config"] * cost_config + \
               env._weights["w_conn"] * data_transfer_cost_normalized + \
               env._weights["w_energy_SEW"] * energy_cost_normalized_SEW + env._weights["w_energy_Phone"] \
               * energy_cost_normalized_Phone
        self._previous_cost = cost

        self._logger.log("Total system cost --> {}".format(cost), 3)
        # restore indentation level for logging
        self._logger.level -= 2

        env._reward = -cost
        env._energy_SEW = energy_cost_normalized_SEW
        env._energy_Phone = energy_cost_normalized_Phone

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
        info["chosen_env"] = self.chosen_env
        '''info["configuration"] = self._current_configuration
        info["battery"] = self._system.devices[0]._get_battery(
            self._current_step)
        info["l_cloud"] = self._system.devices[2]._get_latency(
            self._current_step, self._current_configuration)
        info["workload"] = self._system.devices[2]._get_workload(
            self._current_step)
        info["wifi"] = self._system.network_domains["WIFI7"]._get_rate(
            self._current_step)
        info["5G"] = self._system.network_domains["5G"]._get_rate(
            self._current_step)'''
        info["envs"] = self._envs

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
                            'Initial value for state elt {} not provided; using default ({})'.format(el_name,
                                                                                                     default_value))
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
        config = self._envs[self.selected_env].configurations[self.selected_action - 1]
        r_wifi = state[0] * (self._elements_max[0] - self._elements_min[0]) + self._elements_min[0]
        r_5G = state[1] * (self._elements_max[1] - self._elements_min[1]) + self._elements_min[1]
        l_SEW = config.partitions[0].latency
        l_phone = config.partitions[1].latency
        l_cloud = config.partitions[2].latency
        w_exec = state[5]
        w_conn = state[6]
        w_config = state[7]
        w_energy_Phone = state[8]
        w_energy_SEW = state[9]
        threshold = state[10] * (self._elements_max[10] - self._elements_min[10]) + self._elements_min[10]

        return [r_wifi,r_5G,l_SEW,l_phone,l_cloud,w_exec,w_conn,w_config,w_energy_Phone,w_energy_SEW,threshold]


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
        s0 = (action % (self._discretization_value + 1)) / self._discretization_value
        s1 = (action // (self._discretization_value + 1)) / self._discretization_value
        # for each state element compute value_norm = (original_value - min )/ (max - min)
        # if min = 0; value_norm = original_value / max
        # then it value is rounded to x decimals to have  discrete values value per state feature, x is determined form the define_decimals function
        r_wifi_norm = (state[0] - self._elements_min[0]) / (self._elements_max[0] - self._elements_min[0])
        r_5G_norm = (state[1] - self._elements_min[1]) / (self._elements_max[1] - self._elements_min[1])
        l_SEW_norm = (state[2] - self._elements_min[2]) / (self._elements_max[2] - self._elements_min[2])
        l_phone_norm = (state[3] - self._elements_min[3]) / (self._elements_max[3] - self._elements_min[3])
        l_cloud_norm = (state[4] - self._elements_min[4]) / (self._elements_max[4] - self._elements_min[4])
        w_exec = state[5]
        w_conn = state[6]
        w_config = state[7]
        w_energy_Phone = state[8]
        w_energy_SEW = state[9]
        threshold_norm = (state[10] - self._elements_min[10]) / (self._elements_max[10] - self._elements_min[10])

        return [r_wifi_norm,r_5G_norm,l_SEW_norm,l_phone_norm,l_cloud_norm,w_exec,w_conn,w_config,w_energy_Phone,w_energy_SEW,threshold_norm]


    def _init_single_env(self,weight,tmax,env_number,initial_sol,initial_sol_file,system_file):
        np.random.seed(self._seed*env_number)
        random.seed(self._seed*env_number)
        env_single = EnvSingle(self._logger)
        _system = System(config_file=system_file,
                            logger=Logger(stream=self._logger.stream,
                                        verbose=self._logger.verbose,
                                        level=self._logger.level + 1), variations=self._varying_parameters,env_number=env_number,max_env_step=self._max_env_step)

        env_single._tmax = tmax
        env_single._system = _system
        # define initial configuration
        self._logger.log(f"Setting the initial solution for env with weight: {str(weight)} and T_max: {str(tmax)} ", 2)
        if initial_sol is not None:
            self._initial_solution = initial_sol
        elif initial_sol_file is not None:
            self._initial_solution = Solution(_system,
                                            initial_sol_file=initial_sol_file,
                                            logger=Logger(stream=self._logger.stream,
                                                            verbose=self._logger.verbose, level=self._logger.level + 1))
        else:
            self._logger.log("No initial solution available", 2)
            self._initial_solution = Solution(_system,
                                            logger=Logger(stream=self._logger.stream,
                                                            verbose=self._logger.verbose,
                                                            level=self._logger.level + 1))
        env_single._current_configuration = self._initial_solution.configuration

        self._initial_solution = None
        env_single._weights = {}
        env_single._weights["w_energy_SEW"] = weight["w_energy_SEW"]
        env_single._weights["w_energy_Phone"] = weight["w_energy_Phone"]
        env_single._weights["w_exec"] = weight["w_exec"]
        env_single._weights["w_config"] = weight["w_config"]
        env_single._weights["w_conn"] = weight["w_conn"]
        if env_number == 1:
            self._weights["w_energy_SEW"] = weight["w_energy_SEW"]
            self._weights["w_energy_Phone"] = weight["w_energy_Phone"]
            self._weights["w_exec"] = weight["w_exec"]
            self._weights["w_config"] = weight["w_config"]
            self._weights["w_conn"] = weight["w_conn"]
        env_single._previous_state = None
        env_single._reward = -1.5
        env_single._previous_action = None
        initial_state = copy.deepcopy(self._initial_state)
        for _ in range(6):
            initial_state.pop(-1)
        env_single._initial_state = copy.deepcopy(initial_state)
        env_single._initial_state.append(weight["w_exec"])
        env_single._initial_state.append(weight["w_conn"])
        env_single._initial_state.append(weight["w_config"])
        env_single._initial_state.append(weight["w_energy_Phone"])
        env_single._initial_state.append(weight["w_energy_SEW"])
        env_single._initial_state.append(tmax)
        self._logger.log(f"env init sate {env_single._initial_state}",2)

        return env_single
    
    def _step_in_env(self, env_, chosen_action):
        env = env_
        env._current_step +=1
        self._logger.log(f"ENV CURRENT STEP {env._current_step}",1)
        if env._true_action is not None:
            if int(chosen_action) != int(env._true_action.id):
                env._current_configuration = self._actions[str(
                    chosen_action)].apply(env._system.configurations)
                env._previous_action = self._actions[str(
                    chosen_action)]
                env._true_action = self._actions[str(chosen_action)]
            else:
                env._previous_action = self._actions["0"]
        else:
            env._current_configuration = self._actions[str(
                chosen_action)].apply(env._system.configurations)
            env._previous_action = self._actions[str(
                chosen_action)]
            env._true_action = self._actions[str(chosen_action)]

        self._logger.log(f"ENV CURRENT STEP {env._current_step}",1)
        r_WIFI = env._system.network_domains["WIFI7"]._get_rate(env._current_step)
        r_5G = env._system.network_domains["5G"]._get_rate(env._current_step)
        #battery = self._system.devices[0]._get_battery(self._current_step)["value"]
        

        l_SEW = env._current_configuration.partitions[0].latency
        l_phone = env._current_configuration.partitions[1].latency
        l_cloud = env._system.devices[2]._get_latency(env._current_step, env._current_configuration)

        if not self._evaluation:
            w_exec = env._weights["w_exec"]
            w_conn = env._weights["w_conn"]
            w_config = env._weights["w_config"]
            w_energy_Phone = env._weights["w_energy_Phone"]
            w_energy_SEW = env._weights["w_energy_SEW"]
        else:
            w_exec = self._weights["w_exec"]
            w_conn = self._weights["w_conn"]
            w_config = self._weights["w_config"]
            w_energy_Phone = self._weights["w_energy_Phone"]
            w_energy_SEW = self._weights["w_energy_SEW"]

        threshold = env._tmax
        env._wifi = r_WIFI
        env._5G = r_5G
        env._l_cloud = l_cloud

        env._previous_state = [r_WIFI,r_5G,l_SEW,l_phone,l_cloud,w_exec,w_conn,w_config,w_energy_Phone,w_energy_SEW,threshold]


        return env

EnvironmentSEWDQNExtended.register()
