import pdb
import sys
import inspect
import torch.optim as optim
import torch.nn.functional as F
import random
from utils import get_current_timestamp, convert_state_to_index, save_checkpoint, load_checkpoint
from mushroom_rl.utils.parameters import ExponentialParameter, Parameter
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.callbacks import CollectDataset, CollectMaxQ, CollectQ
from mushroom_rl.approximators.parametric.torch_approximator import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.environments import *
from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.value import *
from joblib import Parallel, delayed, parallel_backend
from classes.Env import EnvironmentSEWQ
from classes.Env_DQN import EnvironmentSEWDQN
from classes.Env_DQN_Extended import EnvironmentSEWDQNExtended
from classes.Core import CoreContinue
from classes.Network import Network
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from classes.Logger import Logger as Logger1
from datetime import datetime
import os
import json
import argparse
import re
import math
import dill as pickle
from zipfile import ZipFile

matplotlib.use('Agg')

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True

def continue_seed(state_random, state_np):
    random.setstate(state_random)
    np.random.set_state(state_np)


def get_uncompleted_directories(Dir):
    list_uncompleted_dir = []
    list_completed_dir = []
    dirs = [x[0] for x in os.walk(Dir)]
    for directory in dirs:
        saved_step = None
        setup_json = directory + "/setup.json"
        if os.path.exists(setup_json):
            with open(setup_json) as f:
                setup = json.load(f)
            n_steps = setup["n_steps"]
            for item in os.listdir(directory):
                if item.endswith(".zip") or item.endswith(".pickle"):
                    s = [float(s) for s in re.findall(r'-?\d+\.?\d*', item)]
                    if len(s) > 0:
                        saved_step = int(s[0])
                        break
            if saved_step:
                if saved_step < n_steps:
                    list_uncompleted_dir.append(directory)
                else:
                    list_completed_dir.append(directory)

    return list_uncompleted_dir, list_completed_dir

def get_core_params(env_params_files, system_files, initial_configs):
    
    core_params = []

    for (env_params_file, system_file, initial_config) in zip(env_params_files, system_files, initial_configs):
        input_file_names = [system_file, initial_config, env_params_file]
        with open(env_params_file) as f:
            params = json.load(f)
        if "DQNParameters" in params:
            DQNParameters = params["DQNParameters"]
        else:
            DQNParameters = {}
        state_discretization = params["EnvParameters"]["StateDiscretization"]
        Setup = params["Setup"]
        methods = Setup["Methods"]
        extended_state = Setup["extended_state"]
        steps_per_env = Setup["steps_per_env"]
        names_scenario = Setup["names_scenario"]
        variations = Setup["names_variation"]
        names_lrs = Setup["learning_strategy"]
        N_steps = Setup["n_steps"]
        n_experiment = Setup["n_experiment"]
        #n_steps = [N_steps] * n_experiment
        e = Setup["epsilon"]
        step_per_checkpoint = Setup["step_per_checkpoint"]
        t_max = Setup["t_max"]
        all_steps_per_env = Setup["all_steps_per_env"]
        round_robin_policy = Setup["round_robin_policy"]
        evaluation = Setup["evaluation"]
        initial_state = []
        if "initial_state" in Setup and Setup["initial_state"]:
            initial_state.append(Setup["initial_state"]["r_wifi"])
            initial_state.append(Setup["initial_state"]["r_5G"])
            initial_state.append(Setup["initial_state"]["battery"])
            initial_state.append(Setup["initial_state"]["l_SEW"])
            initial_state.append(Setup["initial_state"]["l_phone"])
            initial_state.append(Setup["initial_state"]["l_cloud"])
        with open(system_file, 'r') as f:
            data = json.load(f)
        data["Constraints"]["T_max"] = t_max
        system = json.dumps(data, indent=2)
        with open(system_file, "w") as f:
            f.write(system)
        weights = []
        Methods = []
        for method in methods:
            if method in globals():
                Methods.append(globals()[method])
            else:
                error_json.log("Method {} does not exist in mushroom".format(method))
                sys.exit(1)

        currentDateAndTime = datetime.now()
        currentTime = currentDateAndTime.strftime("%d_%m_%Y_%H_%M_%S")
        directory = os.path.dirname(os.path.abspath(env_params_file)) + "/logs/{}".format(currentTime)
        if not os.path.exists(directory):
            os.makedirs(directory)
        lr_exp = 0
        for variation in variations:
            directory1 = directory + "/" + '_'.join(variation)
            if not os.path.exists(directory1):
                os.mkdir(directory1)
            for learning_strategy in names_lrs:
                if learning_strategy == "ConstantLr":
                    lr = Setup["lr"]
                elif learning_strategy == "decayingLr":
                    lr = 1
                    lr_exp = Setup["exp"]
                else:
                    error_json.log("Learning strategy should be either 'ConstantLr' or 'decayingLr'.")
                    sys.exit(1)
                directory2 = directory1 + "/" + learning_strategy
                if not os.path.exists(directory2):
                    os.mkdir(directory2)
                if not extended_state:
                    for scenario in names_scenario:
                        directory3 = directory2 + "/" + scenario
                        if not os.path.exists(directory3):
                            os.mkdir(directory3)
                        if scenario == "Var_Tmax":
                            t_max = names_scenario[scenario]
                        elif scenario == "Var_Weights":
                            weights = names_scenario[scenario]
                        else:
                            error_json.log("Scenario should be either 'Var_Tmax' or 'Var_Weights'.")
                            sys.exit(1)
                        for a in Methods:
                            directory4 = directory3 + "/" + a.__name__
                            if not os.path.exists(directory4):
                                os.mkdir(directory4)
                            for i in range(n_experiment):

                                core_params.append((a, e, i+1, variation, scenario, lr, lr_exp, weights,
                                                    t_max, step_per_checkpoint, N_steps, DQNParameters,
                                                    input_file_names,state_discretization, directory4,extended_state,steps_per_env,all_steps_per_env,round_robin_policy,evaluation))
                else:
                    if "Var_Tmax" in names_scenario and "Var_Weights" in names_scenario:
                        t_max = names_scenario["Var_Tmax"]
                        weights = names_scenario["Var_Weights"]
                        for a in Methods:
                            directory4 = directory2 + "/" + a.__name__
                            if not os.path.exists(directory4):
                                os.mkdir(directory4)
                            for i in range(n_experiment):

                                core_params.append((a, e, i+1, variation, "all", lr, lr_exp, weights,
                                                    t_max, step_per_checkpoint, N_steps, DQNParameters,
                                                    input_file_names,state_discretization, directory4,extended_state,steps_per_env,all_steps_per_env,round_robin_policy,evaluation))
                    else:
                        error_json.log("Scenario should be have 'Var_Tmax' and 'Var_Weights' in case of extended_state = true.")
                        sys.exit(1)

    return core_params

def experiment(core_param):
    
    algorithm_class = core_param[0]
    eps = core_param[1]
    n_exp = core_param[2]
    variation = core_param[3]
    scenario = core_param[4]
    lr = core_param[5]
    lr_exp = core_param[6]
    weights = core_param[7]
    t_max = core_param[8]
    step_per_checkpoint = core_param[9]
    n_steps = core_param[10]
    DQNParameters = core_param[11]
    input_file_names = core_param[12]
    state_discretization = core_param[13]
    directory4 = core_param[14]
    extended_state = core_param[15]
    steps_per_env = core_param[16]
    all_steps_per_env = core_param[17]
    round_robin_policy = core_param[18]
    evaluation = core_param[19]
    file_name = core_param[20]
    if file_name is not None:
        file_stream = open(file_name,"w")
    else:
        file_stream = sys.stdout

    seed = n_exp * 1234
    setup_seed(seed)
    q=[]
    # MDP
    Dir = directory4 + "/exp_{}_{}".format(n_exp, algorithm_class.__name__)
    if not os.path.exists(Dir):
        os.mkdir(Dir)
    logger = Logger(algorithm_class.__name__, results_dir=Dir,log_console=False)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + algorithm_class.__name__)
    logger.info('Epsilon: {}'.format(eps))
    logger.info('Alg: {}'.format(algorithm_class.__name__))


    filename = Dir + "/training_{}.txt".format(n_steps)
    setup = {}
    setup["method"] = algorithm_class.__name__
    setup["names_variation"] = variation
    setup["StateDiscretization"] = state_discretization
    if lr ==1:
        setup["learning_strategy"] = "decayingLr"
        setup["exp"] = lr_exp
    else:
        setup["learning_strategy"] = "ConstantLr"
    setup["n_steps"] = n_steps
    setup["n_experiment"] = n_exp
    setup["epsilon"] = eps
    setup["step_per_checkpoint"] = step_per_checkpoint
    setup["t_max"] = t_max
    setup["extended_state"] = extended_state
    setup["round_robin_policy"] = round_robin_policy
    setup["evaluation"] = evaluation
    setup["all_steps_per_env"] = all_steps_per_env
    setup["steps_per_env"] = steps_per_env
    if not extended_state:
        if scenario == "Var_Tmax":
            setup["name_scenario"] = {"Var_Tmax": t_max}
        else:
            setup["name_scenario"] = {"Var_Weights": weights}
    else:
        setup["name_scenario"] = {"Var_Weights": weights,"Var_Tmax": t_max}
    
    setup["step_per_checkpoint"] = step_per_checkpoint
    setup_json = Dir + "/setup.json"

    

    pi = EpsGreedy(epsilon=eps)
    learning_rate = Parameter(value=lr)
    qt = []
    if "DQN" in algorithm_class.__name__:
        setup["DQNParameters"] = DQNParameters
        if DQNParameters:
            if not extended_state:
                mdp = EnvironmentSEWDQN(system_file=input_file_names[0],
                                        initial_sol_file=input_file_names[1],
                                        env_params_file=input_file_names[2],
                                        logger=Logger1(verbose=3), variations=variation)
            else:
                

                mdp = EnvironmentSEWDQNExtended(system_file=input_file_names[0],
                                        initial_sol_file=input_file_names[1],
                                        env_params_file=input_file_names[2],
                                        logger=Logger1(stream=file_stream,verbose=3), variations=variation,weights=weights,tmaxs=t_max,seed=seed,steps_per_env=steps_per_env,all_steps_per_env=all_steps_per_env,round_robin=round_robin_policy,evaluation=evaluation)

            # DQN Parameters
            try:
                n_steps_per_fit = DQNParameters["n_steps_per_fit"]
                initial_replay_size = DQNParameters["initial_replay_size"]
                max_replay_size = DQNParameters["max_replay_size"]
                target_update_frequency = DQNParameters["target_update_frequency"]  # every (n_steps_per_fit * target_update_frequency) steps, the target network weights will be updated (fit function in abstract_dqn.py)
                batch_size = DQNParameters["batch_size"]
                use_cuda = DQNParameters["use_cuda"]
                #train_frequency = 1


                # Approximator
                n_features = DQNParameters["approximator_params"]["n_features"]
                fun_layers = DQNParameters["approximator_params"]["fun_layers"]
                optimizer = DQNParameters["approximator_params"]["optimizer"]
                fun = DQNParameters["approximator_params"]["loss"]
                opt_class = optimizer["class"]
                DNN_lr = optimizer["params"]["lr"]
                dropout = DQNParameters["approximator_params"]["dropout"]
                dropout_list = []
                if dropout:
                    if "dropout_list" in DQNParameters["approximator_params"] and \
                            len(DQNParameters["approximator_params"]["dropout_list"]) == len(n_features):
                        dropout_list = DQNParameters["approximator_params"]["dropout_list"]
                    else:
                        logger.error("'dropout_list' (with the same length as 'n_features' list) should be provided in config json file")
                        sys.exit(1)
                funs = [item for item in inspect.getmembers(F, inspect.isfunction) if fun in item]
                if len(funs)>0:
                    loss_fun = funs[0][1]
                else:
                    logger.error("loss function {} does not exist".format(fun))
                    sys.exit(1)
                classes = [item for item in inspect.getmembers(optim, inspect.isclass) if opt_class in item]
                if len(classes)>0:
                    optim_class = classes[0][1]
                else:
                    logger.error("loss function {} does not exist".format(opt_class))
                    sys.exit(1)
            except Exception as err:
                logger.error("Filed with '{}' name does not exist in DQNParameters section of config json file".format(err))
                sys.exit(1)

            input_shape = mdp.info.observation_space.shape
            approximator_params = dict(network=Network,
                                    optimizer={'class': optim_class,
                                                'params': {'lr': DNN_lr}},
                                    loss=loss_fun,
                                    input_shape=input_shape,
                                    n_features=n_features,
                                    fun_layers=fun_layers,
                                    use_cuda=use_cuda,
                                    output_shape=mdp.info.action_space.size,
                                    n_actions=mdp.info.action_space.n,
                                    dropout=dropout,
                                    dropout_list=dropout_list,
                                    seed=seed)
            print("agent here 1")
            agent = algorithm_class(mdp.info, pi, 
                        approximator_params=approximator_params, approximator=TorchApproximator, batch_size=batch_size,
                        initial_replay_size=initial_replay_size,
                        max_replay_size=max_replay_size,
                        target_update_frequency=target_update_frequency)
            
            agent.set_logger(logger)
            start = mdp.normalize_state(mdp._initial_state)
            

        else:
            logger.error("DQN parameters are not specified")
            sys.exit(1)
        collect_max_Q = CollectMaxQ(agent.approximator, np.array(start))
        collect_dataset = CollectDataset()
        collect_Q = CollectQ(agent.approximator)
    else:
        n_steps_per_fit = 1
        mdp = EnvironmentSEWQ(system_file=input_file_names[0],
                            initial_sol_file=input_file_names[1],
                            env_params_file=input_file_names[2],
                            logger=Logger1(verbose=3), variations=variation)
        if lr == 1:
            learning_rate = ExponentialParameter(value=1, exp=lr_exp, max_value=1., min_value=0.0001, size=mdp.info.size)

        algorithm_params = dict(learning_rate=learning_rate)
        agent = algorithm_class(mdp.info, pi, **algorithm_params)
        agent.set_logger(logger)

        start = convert_state_to_index(mdp.normalize_state(mdp._initial_state), state_discretization+1)

        collect_max_Q = CollectMaxQ(agent.Q, np.array([start]))
        collect_dataset = CollectDataset()
        collect_Q = CollectQ(agent.Q)
    system = json.dumps(setup, indent=2)
    with open(setup_json, "w") as f:
        f.write(system)
    if scenario == "Var_Tmax":
        n_env_variation = len(t_max)
    else:
        n_env_variation = len(weights)
    inter = n_steps//n_env_variation
    list_interval = list(range(1, n_steps, inter))
    def callback_step(step):
        mdp._logger.log("Step " + str(mdp._current_step) + " :",1)

        #changing epsilon
        '''exponent = min(1, mdp._current_step / (n_steps//3))
        epsilon = 1 * math.pow(eps, exponent)
        mdp._logger.log("epsilon " + str(epsilon),1)
        agent.policy._epsilon = Parameter(epsilon)'''

        loss = agent.approximator.model.loss_fit
        mdp._logger.log("loss " + str(loss),1)
        if (not extended_state) or evaluation:
            for idx, i in enumerate(list_interval):

                if mdp._current_step == i:
                    if "DQN" not in agent.__class__.__name__:
                        learning_rate = ExponentialParameter(value=1, exp=lr_exp, max_value=1., min_value=0.0001, size=mdp.info.size)
                        agent._alpha = learning_rate
                    if scenario == "Var_Tmax":
                        if idx >= len(t_max):
                            mdp._tmax = t_max[-1]
                        else:
                            mdp._tmax = t_max[idx]
                    elif scenario == "Var_Weights":
                        if idx >= len(weights):
                            mdp._weights["w_energy_SEW"] = weights[-1]["w_energy_SEW"]
                            mdp._weights["w_energy_Phone"] = weights[-1]["w_energy_Phone"]
                            mdp._weights["w_exec"] = weights[-1]["w_exec"]
                            mdp._weights["w_config"] = weights[-1]["w_config"]
                            mdp._weights["w_conn"] = weights[-1]["w_conn"]
                        else:
                            mdp._weights["w_energy_SEW"] = weights[idx]["w_energy_SEW"]
                            mdp._weights["w_energy_Phone"] = weights[idx]["w_energy_Phone"]
                            mdp._weights["w_exec"] = weights[idx]["w_exec"]
                            mdp._weights["w_config"] = weights[idx]["w_config"]
                            mdp._weights["w_conn"] = weights[idx]["w_conn"]
                    if evaluation:
                        if idx >= len(weights):
                            mdp._weights["w_energy_SEW"] = weights[-1]["w_energy_SEW"]
                            mdp._weights["w_energy_Phone"] = weights[-1]["w_energy_Phone"]
                            mdp._weights["w_exec"] = weights[-1]["w_exec"]
                            mdp._weights["w_config"] = weights[-1]["w_config"]
                            mdp._weights["w_conn"] = weights[-1]["w_conn"]
                        else:
                            mdp._weights["w_energy_SEW"] = weights[idx]["w_energy_SEW"]
                            mdp._weights["w_energy_Phone"] = weights[idx]["w_energy_Phone"]
                            mdp._weights["w_exec"] = weights[idx]["w_exec"]
                            mdp._weights["w_config"] = weights[idx]["w_config"]
                            mdp._weights["w_conn"] = weights[idx]["w_conn"]
        

        max_diff_Qs = 0
        if agent.__class__.__name__ == "DoubleQLearning":
            max_diff_Qs = np.float64(abs(agent.Q[0].table - agent.Q[1].table).max())
        if "DQN" in agent.__class__.__name__:
            max_diff_Qs = np.float64(abs(agent.approximator.predict(step[0][0]) - agent.target_approximator.predict(step[0][0])).max())
        mdp._save_info_to_file(
            mdp._get_info(), {"latency": mdp._latency,'T_max': mdp._tmax,"energy":mdp._energy, "max_diff_Qs": max_diff_Qs,
                            "energy_phone":mdp._energy_Phone,"sel_action":mdp.selected_action,"loss":loss}, filename)

        #if mdp._current_step == 900:
        #   sys.exit(1)

    def callback_fit(step):
        mdp._logger.log("Step fit " + str(mdp._current_step) + " :",1)

        #changing epsilon
        '''exponent = min(1, mdp._current_step / (n_steps//3))
        epsilon = 1 * math.pow(eps, exponent)
        mdp._logger.log("epsilon " + str(epsilon),1)
        agent.policy._epsilon = Parameter(epsilon)'''

        loss = agent.approximator.model.loss_fit
        mdp._logger.log("loss " + str(loss),1)

        max_diff_Qs = 0
        if agent.__class__.__name__ == "DoubleQLearning":
            max_diff_Qs = np.float64(abs(agent.Q[0].table - agent.Q[1].table).max())
        if "DQN" in agent.__class__.__name__:
            max_diff_Qs = np.float64(abs(agent.approximator.predict(step[0][0]) - agent.target_approximator.predict(step[0][0])).max())
        mdp._save_info_to_file(
            mdp._get_info(), {"latency": mdp._latency,'T_max': mdp._tmax,"energy":mdp._energy, "max_diff_Qs": max_diff_Qs,
                            "energy_phone":mdp._energy_Phone,"sel_action":mdp.selected_action,"loss":loss}, filename)

        #if mdp._current_step == 900:
        #   sys.exit(1)



    def collect_last_Q(sample):
        if mdp._current_step == n_steps - 1:
            collect_Q.get()
    def collect_checkpoint(sample):
        for samp in sample:
            current_q = agent.policy._approximator.predict(samp[0])[samp[1]][0]
            q.append(current_q)
            #print("Q here ", current_q)
        state_random = random.getstate()
        state_np = np.random.get_state()
        steps = n_steps if not all_steps_per_env else n_steps*len(mdp._envs)
        save_checkpoint(agent, mdp, core, collect_dataset, collect_max_Q, collect_Q, steps, state_random, state_np, step_per_checkpoint, q, Dir)

    callbacks = [collect_dataset, collect_max_Q, collect_checkpoint]

    core = Core(agent, mdp, callbacks, callback_step)

    if evaluation:
        print("Loading DNN state...")
        pickle_file = "checkpoints/list_params_{}.pickle".format(n_steps)
        list_params = []
        with open(pickle_file, "rb") as fp:  # Unpickling
            list_params = pickle.load(fp)
        
        zip_file = "checkpoints/agent_{}.zip".format(n_steps)
        with ZipFile(zip_file, 'r') as file:
            agent = Serializable.load_zip(file)
        agent.approximator.model.network.set_state(list_params[11])
        #agent.policy.set_epsilon(Parameter(0))
        core.agent = agent
   
    mdp._agent = agent
    # Train
    steps = n_steps if not all_steps_per_env else n_steps*len(mdp._envs)
    if not evaluation:
        core.learn(n_steps=steps, n_steps_per_fit=n_steps_per_fit, quiet=False, render=False)
    else:
        core.learn(n_steps=steps, n_steps_per_fit=n_steps_per_fit, quiet=False, render=False)  
        #core.evaluate(n_steps=steps, quiet=False, render=False)

    _, _, reward, _, _, _ = parse_dataset(collect_dataset.get())
    #reward=[0]*n_steps
    max_Qs = collect_max_Q.get()

    qt = np.array(q)
    if file_name is not None:
        file_stream.close()

    return reward, max_Qs, n_steps, qt, (os.path.dirname(Dir), n_exp)

def experiment_resume(Dir):
    setup_json = Dir + "/setup.json"
    with open(setup_json) as f:
        setup = json.load(f)
    weights = []
    scenario = setup["name_scenario"]
    state_discretization = setup["StateDiscretization"]
    n_exp = setup["n_experiment"]
    method = setup["method"]
    learning_strategy = setup["learning_strategy"]
    if learning_strategy == "decayingLr":
        lr_exp = setup["exp"]
    t_max = setup["t_max"]
    if "Var_Tmax" in scenario.keys():
        t_max = setup["name_scenario"]["Var_Tmax"]
    else:
        weights = setup["name_scenario"]["Var_Weights"]
    step_per_checkpoint = setup["step_per_checkpoint"]
    n_steps = setup["n_steps"]

    mdp, agent, current_step, list_params, dataset, max_Q, Q = load_checkpoint(Dir, n_steps)
    continue_seed(list_params[8], list_params[9])
    q = list_params[10]
    if "DQN" in method:
        n_steps_per_fit = setup["DQNParameters"]["n_steps_per_fit"]
        collect_max_Q = CollectMaxQ(agent.approximator, mdp._state)
        collect_Q = CollectQ(agent.approximator)
        agent.approximator.model.network.set_state(list_params[11])
    else:
        n_steps_per_fit = 1
        start = convert_state_to_index(mdp.normalize_state(mdp._state), state_discretization+1)
        collect_max_Q = CollectMaxQ(agent.Q, np.array([start]))
        collect_Q = CollectQ(agent.Q)
    collect_Q._data_list = Q
    collect_dataset = CollectDataset()
    collect_max_Q._data_list = max_Q
    collect_dataset._data_list = dataset

    filename = Dir + "/training_{}.txt".format(n_steps)

    if "Var_Tmax" in scenario.keys():
        n_env_variation = len(t_max)
    else:
        n_env_variation = len(weights)
    inter = n_steps//n_env_variation
    list_interval = list(range(1, n_steps, inter))
    def callback_step(step):
        mdp._logger.log("Step " + str(mdp._current_step) + " :",1)
        for idx, i in enumerate(list_interval):

            if mdp._current_step == i:
                if "DQN" not in agent.__class__.__name__:
                    learning_rate = ExponentialParameter(value=1, exp=lr_exp, max_value=1., min_value=0.0001, size=mdp.info.size)
                    agent._alpha = learning_rate
                if "Var_Tmax" in scenario.keys():
                    if idx >= len(t_max):
                        mdp._tmax = t_max[-1]
                    else:
                        mdp._tmax = t_max[idx]
                elif "Var_Weights" in scenario.keys():
                    if idx >= len(weights):
                        mdp._weights["w_energy_SEW"] = weights[-1]["w_energy_SEW"]
                        mdp._weights["w_energy_Phone"] = weights[-1]["w_energy_Phone"]
                        mdp._weights["w_exec"] = weights[-1]["w_exec"]
                        mdp._weights["w_config"] = weights[-1]["w_config"]
                        mdp._weights["w_conn"] = weights[-1]["w_conn"]
                        #mdp.weight_state = 1
                    else:
                        mdp._weights["w_energy_SEW"] = weights[idx]["w_energy_SEW"]
                        mdp._weights["w_energy_Phone"] = weights[idx]["w_energy_Phone"]
                        mdp._weights["w_exec"] = weights[idx]["w_exec"]
                        mdp._weights["w_config"] = weights[idx]["w_config"]
                        mdp._weights["w_conn"] = weights[idx]["w_conn"]
                        '''if len(weights)>1:
                            mdp.weight_state = idx / len(weights)-1
                        else:
                            mdp.weight_state = 0'''
        max_diff_Qs = 0
        if agent.__class__.__name__ == "DoubleQLearning":
            max_diff_Qs = np.float64(abs(agent.Q[0].table - agent.Q[1].table).max())
        if "DQN" in agent.__class__.__name__:
            max_diff_Qs = np.float64(abs(agent.approximator.predict(step[0][0]) - agent.target_approximator.predict(step[0][0])).max())

        mdp._save_info_to_file(
            mdp._get_info(), {"latency": mdp._latency, 'T_max': mdp._tmax,"energy":mdp._energy, "max_diff_Qs": max_diff_Qs,
                              "energy_phone":mdp._energy_Phone,"sel_action":mdp.selected_action}, filename)



    def collect_last_Q(sample):
        if mdp._current_step == n_steps - 1:
            collect_Q.get()

    def collect_checkpoint(sample):
        for samp in sample:
            current_q = agent.policy._approximator.predict(samp[0])[samp[1]][0]
            q.append(current_q)
            
        state_random = random.getstate()
        state_np = np.random.get_state()
        save_checkpoint(agent, mdp, core, collect_dataset, collect_max_Q, collect_Q, n_steps, state_random, state_np, step_per_checkpoint, q, Dir)

    callbacks = [collect_dataset, collect_max_Q, collect_last_Q, collect_checkpoint]

    core = CoreContinue(agent, mdp, callbacks, callback_step, list_params)

    core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, quiet=True, render=False)

    _, _, reward, _, _, _ = parse_dataset(collect_dataset.get())
    max_Qs = collect_max_Q.get()

    qt = np.array(q)

    return reward, max_Qs, n_steps, qt, (os.path.dirname(Dir), n_exp)

def experiment_completed(Dir):
    setup_json = Dir + "/setup.json"
    with open(setup_json) as f:
        setup = json.load(f)
    method = setup["method"]
    n_exp = setup["n_experiment"]
    state_discretization = setup["StateDiscretization"]
    n_steps = setup["n_steps"]
    mdp, agent, current_step, list_params, dataset, max_Q, Q, q = load_checkpoint(Dir, n_steps)
    q = list_params[10]
    if "DQN" in method:
        collect_max_Q = CollectMaxQ(agent.approximator, mdp._state)
        collect_Q = CollectQ(agent.approximator)
        agent.approximator.model.network.set_state(list_params[11])
    else:
        start = convert_state_to_index(mdp.normalize_state(mdp._state), state_discretization+1)
        collect_max_Q = CollectMaxQ(agent.Q, np.array([start]))
        collect_Q = CollectQ(agent.Q)
    collect_Q._data_list = Q
    collect_dataset = CollectDataset()
    collect_max_Q._data_list = max_Q
    collect_dataset._data_list = dataset

    _, _, reward, _, _, _ = parse_dataset(collect_dataset.get())
    max_Qs = collect_max_Q.get()

    qt = np.array(q)

    return reward, max_Qs, n_steps, qt, (os.path.dirname(Dir), n_exp)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="PMAIEDGE")
    parser.add_argument("-A", "--Apps_dir",
                        help="The applications directory")
    parser.add_argument("--start_from_checkpoint", dest='start_from_checkpoint', default=False,
                        action='store_true',  help="Start from checkpoint")
    parser.add_argument("-D", "--data_directory",
                        help="Directory which includes checkpoints")
    parser.add_argument("-l", "--logfile",
                        help="logfile where to print the logs")
    args = parser.parse_args()

    # initialize error stream
    error_json = Logger1(stream=sys.stderr, verbose=1, is_error=True)
    # check if the system configuration file exists
    start_from_checkpoint = args.start_from_checkpoint
    if args.start_from_checkpoint:
        if not os.path.exists(args.data_directory):
            error_json.log("Checkpoint directory {} does not exist".format(args.data_directory))
            sys.exit(1)
        else:
            Dir = args.data_directory
    #start_from_checkpoint = True
    if args.Apps_dir:
        if not os.path.exists(args.Apps_dir):
            error_json.log("Applications directory {} does not exist".format(args.Apps_dir))
            sys.exit(1)
        else:
            env_params_file = []
            system_file = []
            initial_config =[]
            for dir in os.listdir(args.Apps_dir):
                if os.path.isdir(os.path.join(args.Apps_dir,dir)):
                    env_params_file.append(os.path.join(args.Apps_dir,dir,"env_config.json"))
                    system_file.append(os.path.join(args.Apps_dir,dir,"system.json"))
                    initial_config.append(os.path.join(args.Apps_dir,dir,"initial_config.json"))
    else:
        env_params_file = ["env_config.json"]
        system_file = ["system.json"]
        initial_config = ["initial_config.json"]
    lr_exp = 0
    legend_labels = []
    fig = plt.figure(figsize=(16, 8))
    if not start_from_checkpoint:
        core_params = get_core_params(env_params_file, system_file, initial_config)
        #experiment(core_params[0])
        with parallel_backend('multiprocessing', n_jobs=-1):
            output = Parallel()(delayed(experiment)(core_param+(args.logfile,)) for core_param in core_params)
    else:
        #Dir = "logs/22_09_2023_19_22_20"
        list_uncompleted_dir, list_completed_dir = get_uncompleted_directories(Dir)
        if len(list_uncompleted_dir) == 0 and len(list_completed_dir) == 0:
            error_json.log("There are no completed and uncompleted experiments")
            sys.exit(1)
        #output = list(experiment_resume(list_uncompleted_dir[0]))
        if len(list_uncompleted_dir) > 0:
            with parallel_backend('multiprocessing', n_jobs=-1):
                output = Parallel()(delayed(experiment_resume)(directory) for directory in list_uncompleted_dir)
        else:
            output = []
        if len(list_completed_dir) > 0:
            for completed_dir in list_completed_dir:
                output.append(experiment_completed(completed_dir))
    outs = {}
    for i in output:
        *key, _ = i[-1]
        outs.setdefault(tuple(key), []).append(i)

    all_outs = list(outs.values())
    for out in all_outs:
        r = np.array([o[0] for o in out])
        max_Qs = np.array([o[1] for o in out])
        n_steps = np.array([o[2] for o in out])
        qt = np.array([o[3] for o in out])
        '''qt = []
        for o in out:
            x = np.zeros(len(o[3]))
            for idx, q in enumerate(o[3]):
                x[idx] = q[o[4][idx][0]]
            qt.append(x)'''
        qt = np.convolve(np.mean(qt, 0), np.ones(100) / 100., 'valid')
        '''qss = []
        for o in out:
            x = np.zeros((len(o[3]),len(o[3][0])))
            for idx, q in enumerate(o[3]):
                for idx1, q_action in enumerate(q):
                    x[idx][idx1] = q_action
            qss.append(x)
        qs = np.array(qss)
        qs_convolve = []
        for i in range(len(o[3][0])):
            qs_convolve.append(np.convolve(np.mean(qs[:, :, i], 0), np.ones(100) / 100., 'valid'))'''
        #qs = np.array([o[3] for o in out])
        #q = np.array([o[3][o[4]] for o in out])
        #qt = np.array([o[4] for o in out])
        directory4 = out[0][4][0]
        np.save(directory4 + '/original_r.npy', np.mean(r, 0))
        r = np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid')
        #q_t = np.mean(qt, 0)
        #qt_s = np.convolve(q_t, np.ones(100) / 100., 'valid')
        max_Qs = np.mean(max_Qs, 0)

        plt.subplot(3, 1, 1)
        plt.plot(r, label="reward", color="green")
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(max_Qs, label="Qmax", color='magenta')
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(qt, label="q", color='red')
        plt.legend()
        fig.savefig(directory4 + "/" + 'test.png')



        # print("directory"+mydir)
        np.save(directory4 + '/r.npy', r)
        np.save(directory4 + '/maxQ.npy', max_Qs)
        np.save(directory4 + '/Qtmean.npy', qt)
        #np.save(directory4 + '/Qt.npy', qt)
        #np.save(directory4 + '/Qtmean.npy', q_t)

