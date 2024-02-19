from classes.Logger import Logger
import time
import sys
import json
import os
import shutil
import re
from mushroom_rl.core import Serializable
from zipfile import ZipFile
import dill as pickle


def sanitize_string(string):

    # lowercase_string = string.lower()
    # .replace("_", "")
    return string.replace("_", "")


def generate_keywords_dict(keys):

    keywords_dict = {}
    for key in keys:
        keywords_dict[sanitize_string(key)] = key
    return keywords_dict


def get_current_timestamp():
    # Get the time in seconds
    # since the epoch
    # using time.time() method

    # time_sec = time.time()

    # Get the time in nanoseconds
    # since the epoch
    # using time.time_ns() method
    time_nanosec = time.time_ns()
    return time_nanosec


def state_to_str(elements, state):
    s = '"state":{'
    for i in range(0, len(elements)):
        s += '"{}":{},'.format(elements[i], state[i])
    s = s[:-1]+'}'
    return s


def save_info_to_file(info, file):
    error = Logger(verbose=1, is_error=True)
    if file:

        with open(file, "a") as f:
            f.write(info)

    else:
        error.log("No file provided cannot save info")
        sys.exit(1)


def get_info_str(elements, info):
    s = '"info":{'
    # s += '"timestamp":{},'.format(info["timestamp"])
    s += '"step":{},'.format(info["step"])
    # s += state_to_str(elements, info["state"])+","
    s += str(info["action"])
    if "l_cloud" in info.keys() and "wifi" in info.keys() and "5G" in info.keys():
        s += ","
        # s += '"cost":{},'.format(info["cost"])
        s += '"l_cloud":{},"wifi":{},"5G":{}'.format(info["l_cloud"],
                                                        info["wifi"],
                                                        info["5G"])
    s += f''',"cost":{info["cost"]}'''
    if "chosen_env" in info.keys():
        s += f''',"chosen_env":{info["chosen_env"]}'''
    if "envs" in info.keys():
        ss=',"envs":{'
        i=0
        for env in info["envs"]:
            ss += f'''"env{i}":{env},'''
            i+=1
        ss = ss[:-1]
        ss+="}"
        s +=ss
    s += ',"others":'
    s += json.dumps(info["others"])
    s += '}'

    return s


def load_info_from_file(file):

    with open(file) as f:
        data = json.load(f)
    load_info_json(data)


def load_info_json(info_json):
    error = Logger(verbose=1, is_error=True)
    info = dict()
    if "info" in info_json.keys():
        _info = info_json["info"]
        if "step" in info.keys():
            info["step"] = _info["step"]
        else:
            error.log("step not in the file")
            sys.exit(1)
        if "state" in info.keys():
            info["state"] = _info["state"]
        else:
            error.log("state not in the file")
            sys.exit(1)
        if "action" in info.keys():
            info["action"] = _info["action"]
        else:
            error.log("action not in the file")
            sys.exit(1)
        if "cost" in info.keys():
            info["cost"] = _info["cost"]
        else:
            error.log("cost not in the file")
            sys.exit(1)
        if "configuration" in info.keys():
            info["configuration"] = _info["configuration"]
        else:
            error.log("configuration not in the file")
            sys.exit(1)
        if "battery" in info.keys():
            batt = _info["battery"]
            value = 0
            plugged = False
            if "value" in batt.keys():
                value = batt["value"]
            else:
                error.log("battery percentage value not in the file")
                sys.exit(1)
            if "plugged" in batt.keys():
                plugged = batt["plugged"]
            info["battery"] = {"value": value, "plugged": plugged}
        else:
            error.log("battery not in the file")
            sys.exit(1)
        if "l_cloud" in info.keys():
            info["l_cloud"] = _info["l_cloud"]
        else:
            error.log("l_cloud not in the file")
            sys.exit(1)
        if "workload" in info.keys():
            info["workload"] = _info["workload"]
        else:
            error.log("workload not in the file")
            sys.exit(1)
        if "wifi" in info.keys():
            info["wifi"] = _info["wifi"]
        else:
            error.log("wifi throughput not in the file")
            sys.exit(1)
        if "5G" in info.keys():
            info["5G"] = _info["5G"]
        else:
            error.log("5G throughput not in the file")
            sys.exit(1)
        if "others" in info.keys():
            info["others"] = _info["others"]
        else:
            info["others"] = {}
    else:
        error.log("info not in the file")
        sys.exit(1)
    return info


def convert_index_to_state(index, discrete_values):
    state = []
    while index > 0:
        state.append((index % discrete_values)/10)
        index //= discrete_values
    if len(state) < 6:
        gap = 6-len(state)
        for i in range(0, gap):
            state.append(0)
    state.reverse()
    return state


def convert_state_to_index(digits, discrete_values):
    index = 0
    for i, digit in enumerate(digits[::-1]):
        index += digit*10 * (discrete_values ** i)
    return int(index)


def save_checkpoint(agent, mdp, core, collect_dataset, collect_max_Q, collect_Q, n_steps, state_random, state_np, step_per_checkpoint, q, Dir):
    if mdp._current_step % step_per_checkpoint == 0 or mdp._current_step == n_steps:
        # save the current checkpoint in temporary folder
        temp_Dir = Dir + "/temp"
        if not os.path.exists(temp_Dir):
            os.mkdir(temp_Dir)
        else:
            for item in os.listdir(temp_Dir):
                os.remove(os.path.join(temp_Dir, item))

        zip_file = temp_Dir + "/agent_{}.zip".format(mdp._current_step)
        with ZipFile(zip_file, 'w') as file:
            agent.save_zip(file, full_save=True)
        zip_file = temp_Dir + "/mdp_{}.zip".format(mdp._current_step)
        with ZipFile(zip_file, 'w') as file:
            mdp.save_zip(file, full_save=True)

        list_params = []
        list_params.append(core._total_episodes_counter )
        list_params.append(core._total_steps_counter)
        list_params.append(core._current_episodes_counter)
        list_params.append(core._current_steps_counter)
        list_params.append(core._episode_steps)
        list_params.append(core._n_episodes)
        list_params.append(core._n_steps_per_fit)
        list_params.append(core._n_episodes_per_fit)
        list_params.append(state_random)
        list_params.append(state_np)
        list_params.append(q)

        if "DQN" in agent.__class__.__name__:
            torch_state = agent.approximator.model.network.get_state()
            list_params.append(torch_state)
            '''list_params.append(agent.approximator._loss_filename)
            pickle_file = Dir + "/regressor_logger_{}.pickle".format(mdp._current_step)
            with open(pickle_file, "wb") as fp:  # Pickling
                pickle.dump(agent.approximator._logger, fp)'''

        pickle_file = temp_Dir + "/list_params_{}.pickle".format(mdp._current_step)
        with open(pickle_file, "wb") as fp:  # Pickling
            pickle.dump(list_params, fp)
        pickle_file = temp_Dir + "/collect_dataset_{}.pickle".format(mdp._current_step)
        with open(pickle_file, "wb") as fp:  # Pickling
            pickle.dump(collect_dataset.get(), fp)
        pickle_file = temp_Dir + "/collect_max_Q_{}.pickle".format(mdp._current_step)
        with open(pickle_file, "wb") as fp:  # Pickling
            pickle.dump(collect_max_Q.get(), fp)
        pickle_file = temp_Dir + "/collect_Q_{}.pickle".format(mdp._current_step)
        with open(pickle_file, "wb") as fp:  # Pickling
            pickle.dump(collect_Q.get(), fp)
        print("checkpoint for {} saved successfully ".format(mdp._current_step))

        # remove the previous checkpoint
        test = os.listdir(Dir)
        for item in test:
            if item.endswith(".zip"):
                os.remove(os.path.join(Dir, item))
            if item.endswith(".pickle"):
                os.remove(os.path.join(Dir, item))

        # move the current checkpoint from temp folder to the main folder
        for file_name in os.listdir(temp_Dir):
            # construct full file path
            source = temp_Dir + "/" + file_name
            destination = Dir + "/" + file_name
            # move only files
            if os.path.isfile(source):
                shutil.move(source, destination)

def load_checkpoint(Dir, n_steps):
    error = Logger(verbose=1, is_error=True)
    current_step = 0
    test = os.listdir(Dir)
    for item in test:
        if item.endswith(".zip"):
            s = [float(s) for s in re.findall(r'-?\d+\.?\d*', item)]
            if len(s) > 0:
                current_step = int(s[0])
                break
    if current_step == 0:
        error.log("There is not any checkpoint files to load!")
        sys.exit(1)
    else:
        txt_log = Dir + "/training_{}.txt".format(n_steps)
        if not os.path.exists(txt_log):
            error.log("There is not any txt log files to load!")
            sys.exit(1)
        else:
            with open(txt_log, 'r') as f:
                data = f.read()
            position = data.find('"info:"{{"step":{}'.format(current_step+1))
            if position >= 0:
                new_data = data[0:position]
                with open(txt_log, "w") as f:
                    f.write(new_data)
        zip_file = Dir + "/agent_{}.zip".format(current_step)
        with ZipFile(zip_file, 'r') as file:
            agent = Serializable.load_zip(file)
        zip_file = Dir + "/mdp_{}.zip".format(current_step)
        with ZipFile(zip_file, 'r') as file:
            mdp = Serializable.load_zip(file)
    pickle_file = Dir + "/list_params_{}.pickle".format(current_step)
    with open(pickle_file, "rb") as fp:  # Unpickling
        list_params = pickle.load(fp)
    pickle_file = Dir + "/collect_dataset_{}.pickle".format(current_step)
    with open(pickle_file, "rb") as fp:  # Unpickling
        dataset = pickle.load(fp)
    pickle_file = Dir + "/collect_max_Q_{}.pickle".format(current_step)
    with open(pickle_file, "rb") as fp:  # Unpickling
        max_Q = pickle.load(fp)
    pickle_file = Dir + "/collect_Q_{}.pickle".format(current_step)
    with open(pickle_file, "rb") as fp:  # Unpickling
        Q = pickle.load(fp)

    '''if agent.__class__.__name__ == "DQN":
        pickle_file = Dir + "/regressor_logger_{}.pickle".format(current_step)
        with open(pickle_file, "rb") as fp:  # Unpickling
            regressor_logger = pickle.load(fp)
        agent.approximator._logger = regressor_logger
        agent.approximator._loss_filename = list_params[-1]'''
    return mdp, agent, current_step, list_params, dataset, max_Q, Q

def define_decimals(value):
    #function to define the number of decimal values according to the discretization
    if value >= 0 and value <= 10:
        return 1
    elif value <= 100:
        return 2
    elif value <= 1000:
        return 3
    # Add more conditions for other ranges if needed
    else:
        return None  # Handle values outside of defined ranges
