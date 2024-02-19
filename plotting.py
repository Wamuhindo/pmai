#!/usr/bin/env python
# coding: utf-8
import pdb
import sys
import shutil
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import multiprocessing as mpp
from multiprocessing import Pool
import functools
from joblib import Parallel, delayed
import re
from main_checkpoint import experiment_completed
from collections import Counter
import heapq
file="log.txt"
FONT_SIZE = 30
plt.rcParams.update({'font.size': 14})


def get_core_param(n_ex, methods_list, Path3):
    core_param = []

    for method in os.listdir(Path3):
        if method in methods_list:
            for i in range(1, n_ex+1):
                core_param.append((i, method))
    if len(core_param) < 1:
        print("There is not any data folders related to the methods")
        sys.exit(1)
    return core_param

def load_data(core_param,Path3):
    print("Number " + str(core_param[0]) + " is loading... \n")
    method_Path = os.path.join(Path3, core_param[1])
    loss = []
    path = os.path.join(method_Path, 'exp_{}_{}'.format(core_param[0], core_param[1]), "training_{}.txt".format(max_step))
    path_setup = os.path.join(method_Path, 'exp_{}_{}'.format(core_param[0], core_param[1]),
                        "setup.json")
    if "DQN" in core_param[1]:
        loss = np.load(os.path.join(
            os.path.join(method_Path, 'exp_{}_{}'.format(core_param[0], core_param[1])),
            core_param[1], "loss_Q.npy"))
    if not os.path.exists(path):
        test1 = os.listdir(os.path.join(method_Path, 'exp_{}_{}'.format(core_param[0], core_param[1])))
        for item1 in test1:
            if item1.endswith(".txt"):
                path = os.path.join(method_Path, 'exp_{}_{}'.format(core_param[0], core_param[1]), item1)
                break

    nRowsRead = None  # specify 'None' if want to read whole file
    df = pd.read_csv(path, delimiter='"info:"', nrows=nRowsRead, index_col=False)
    if os.path.exists(path_setup):
        with open(path_setup) as f:
            setup = json.load(f)
    return df, loss,setup, core_param[1]

def plot_loss(result_folder_dir, scenario_name, losses, existent_methods):
    plt.figure(figsize=(16, 8))
    for idx, method in enumerate(existent_methods):
        if len(losses[idx]) > 0:
            transparency = 1 - 0.2*idx
            I = []
            equal_length = all(len(ele) == len(losses[idx][0]) for ele in losses[idx])
            if not equal_length:
                length = min([len(losses[idx][i]) for i in range(len(losses[idx]))])
                for id, loss in enumerate(losses[idx]):
                    losses[idx][id] = loss[0:length]
            ave_loss=np.mean(losses[idx], axis=0)
            if n_steps_per_fit > 1:
                ave_loss = np.repeat(ave_loss, n_steps_per_fit)
            for i in range(initial_replay_size, initial_replay_size+len(ave_loss)):
                I.append(i)

            plt.plot(I, ave_loss, label=method, alpha=transparency)
    plt.xlabel('steps')
    # naming the y axis
    plt.ylabel('DQN loss')
    plt.legend()

    plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_DQNloss.pdf', dpi=200)

def getEnergyConsumptionInterval1(interval, step, max_step, EC):
    _interval = min(interval, max_step - step)
    energy_consumption = [EC[i + step] for i in range(_interval)]

    return sum(energy_consumption)
#fixing the limit of latency violation to have a clear plot otherwise we could have violation of 1oooms that
#will let the figure not be clear for latency such as 40ms 
def shrink(value,limit):
  if value>limit:
    return limit
  else:
    return value

#function to get the cumulative violation. Moving average
def getViolationInterval1(interval,step,max_step,t_max,lt):
    if  step < (max_step//3):
        tmax = t_max[0]
        
    elif step >= (max_step//3) and step < (2*max_step//3):
        tmax = t_max[step+100]

    elif step >= (2*max_step//3):
        tmax = t_max[(2*max_step//3)+100]

    count = len([i for i in lt[(step-1):step+interval] if i > tmax])
    
        
    return (count*100.)/interval

#function to get the cumulative action
def getCumulativeAction(interval,step,max_step,action):
    import pdb
    _interval = min(interval,max_step-step)
    try:
        count_action = [i for i in range(_interval-1) if action[i+step+1]!=action[i+step] and action[i+step+1]!=0 ]
    except:
        pdb.set_trace()
        
    return len(count_action)


def getScenarioName(path):
    var = ""
    scenario = ""
    setup = ""
    
    if "Var_Weights" in path:
        scenario = "Var_Weights"
    elif "Var_Tmax" in path:
        scenario = "Var_Tmax"
        
    if "ConstantLr" in path:
        setup = "ConstantLr"
    else:
        setup = "DecayLr"
        
    if "1Variation" in path:
        var = "5G"
    elif "2Variations" in path:
        var = "5GWIFI"
    elif "3Variations" in path:
        var = "5GWIFICLOUD"
        
    #print("{}{}_{}".format(setup,scenario,var),file=file)
        
    return "{}{}_{}".format(setup,scenario,var)


def countViolation(result_folder_dir, scenario_name, interval, max_step, t_max, latencies, existent_methods):

    counts = []
    plt.figure(figsize=(16, 8))
    for idx, method in enumerate(existent_methods):
        transparency = 1 - 0.2*idx
        I = []
        percentage_method = []
        count_method = []
        for i in range(max_step//interval):
            percentage_ave=[]
            count_ave = []
            for n_exp in range(len(latencies[idx])):
                percentage_ave.append((len([j for j in latencies[idx][n_exp][i * interval:i * interval + interval] if
                      j > t_max[i * interval + 1]]) / interval) * 100)
                count_ave.append(len([j for j in latencies[idx][n_exp][i * interval:i * interval + interval] if
                      j > t_max[i * interval + 1]]))
            percentage_ave_method=np.mean(np.asarray(percentage_ave), 0)
            percentage_method.append(percentage_ave_method)
            count_method.append(np.mean(np.asarray(count_ave), 0))
            I.append(i+1)
        plt.plot(I, percentage_method, label=method, alpha=transparency)
        counts.append(sum(count_method))
    plt.xlabel('steps/' + str(interval))
    # naming the y axis
    plt.ylabel('Violation (%)')
    plt.legend()
    plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_violations_comparision.pdf', dpi=200)

    plt.figure(figsize=(8,8))
    fig, ax=plt.subplots(1,1)
    ax.set_xlim(-0.5, len(existent_methods) - 0.5)
    width = min(1 - 1 / (len(existent_methods) + 1), 0.1)
    bars=plt.bar(existent_methods, counts, width=width)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .1, str(round((yval/max_step)*100,2)) + "%")
    plt.xlabel("Name of methods", fontsize=14)
    plt.ylabel("Total number of violations", fontsize=14)
    plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_total_violations.pdf', dpi=200)

def plot_metrics_seprately_without_range(result_folder_dir, scenario_name,mstart,mstep,x,cum_violation, cum_action,r,
                           qmax, cum_EC, q_diff, qt, method):
    if method == "SARSA":
        color = "orange"
    elif method =="Q":
        color ="blue"
    else:
        color = "green"
    fig, axs = plt.subplots(1, 7, figsize=(20, 30))
    plt.subplots_adjust(hspace=0.3)
    plt.subplot(7, 1, 1)
    plt.plot(x[mstart:mstep + mstart], np.convolve(np.array(r[mstart:mstep + mstart]), np.ones(100) / 100., "same"),
             label=method, color=color)
    plt.legend()
    plt.xlabel('Timestep')
    plt.ylabel('reward')

    plt.subplot(7, 1, 2)
    plt.plot(x[mstart:mstep + mstart], np.convolve(np.array(qmax[mstart:mstep + mstart]), np.ones(1) / 1., "same"),
             label=method, color=color)
    plt.legend()
    plt.xlabel('Timestep')
    plt.ylabel('Q_max')

    plt.subplot(7, 1, 3)
    plt.plot(x[mstart:mstep + mstart-1], np.convolve(np.array(cum_violation[mstart:mstep + mstart-1]), np.ones(1) / 1., "same"), label=method, color=color)
    plt.ylabel('Moving average Violation (%)')
    plt.xlabel('Timestep')
    plt.legend()

    plt.subplot(7, 1, 4)
    plt.plot(x[mstart:mstep + mstart-1], cum_action[mstart:mstep + mstart-1], label=method, color=color)
    # naming the x axis
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel('Moving average number of actions')

    # giving a title to my graph
    plt.legend()

    plt.subplot(7, 1, 5)
    plt.plot(x[mstart:mstep + mstart-1], cum_EC[mstart:mstep + mstart-1], label=method, color=color)
    # naming the x axis
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel('Moving average Energy consumption')
    plt.legend()

    plt.subplot(7, 1, 6)
    plt.plot(qt, label=method, color=color)
    # naming the x axis
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel('q')
    plt.legend()

    if "Double" in method or "DQN" in method:
        plt.subplot(7, 1, 7)
        plt.plot(x[mstart:mstep + mstart], q_diff[mstart:mstep + mstart], label=method, color=color)
        # naming the x axis
        plt.xlabel('Timestep')
        # naming the y axis
        plt.ylabel('differences of qs')
        plt.legend()

    plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_metrics_' + method + '_without_range.pdf', dpi=200)
    print("Metrics of " + method + " plotted and saved successfully!!")

def plot_metrics_seprately(result_folder_dir, scenario_name,mstart,mstep,x,cum_violation, cum_action,r,
                           qmax, cum_EC, q_diff, qt, method,r_range,qmax_range, cum_violation_range,cum_action_range, cum_EC_range, q_diff_range, qt_range):
    if method == "SARSA":
        color = "orange"
    elif method =="Q":
        color ="blue"
    else:
        color = "green"
    fig, axs = plt.subplots(1, 7, figsize=(20, 30))
    plt.subplots_adjust(hspace=0.3)
    plt.subplot(7, 1, 1)
    plt.plot(x[mstart:mstep + mstart], np.convolve(np.array(r[mstart:mstep + mstart]), np.ones(100) / 100., "same"),
             label=method, color=color)
    plt.ylim(r_range[0], r_range[1])
    plt.legend()
    plt.xlabel('Timestep')
    plt.ylabel('reward')

    plt.subplot(7, 1, 2)
    plt.plot(x[mstart:mstep + mstart], np.convolve(np.array(qmax[mstart:mstep + mstart]), np.ones(1) / 1., "same"),
             label=method, color=color)
    plt.ylim(qmax_range[0], qmax_range[1])
    plt.legend()
    plt.xlabel('Timestep')
    plt.ylabel('Q_max')

    plt.subplot(7, 1, 3)
    plt.plot(x[mstart:mstep + mstart-1], np.convolve(np.array(cum_violation[mstart:mstep + mstart-1]), np.ones(1) / 1., "same"), label=method, color=color)
    plt.ylim(cum_violation_range[0], cum_violation_range[1])
    plt.ylabel('Moving average Violation (%)')
    plt.xlabel('Timestep')
    plt.legend()

    plt.subplot(7, 1, 4)
    plt.plot(x[mstart:mstep + mstart-1], cum_action[mstart:mstep + mstart-1], label=method, color=color)
    plt.ylim(cum_action_range[0], cum_action_range[1])
    # naming the x axis
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel('Moving average number of actions')

    # giving a title to my graph
    plt.legend()

    plt.subplot(7, 1, 5)
    plt.plot(x[mstart:mstep + mstart-1], cum_EC[mstart:mstep + mstart-1], label=method, color=color)
    plt.ylim(cum_EC_range[0], cum_EC_range[1])
    # naming the x axis
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel('Moving average Energy consumption')
    plt.legend()

    plt.subplot(7, 1, 6)
    plt.plot(qt, label=method, color=color)
    plt.ylim(qt_range[0], qt_range[1])
    # naming the x axis
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel('q')
    plt.legend()

    if "Double" in method or "DQN" in method:
        plt.subplot(7, 1, 7)
        plt.plot(x[mstart:mstep + mstart], q_diff[mstart:mstep + mstart], label=method, color=color)
        plt.ylim(q_diff_range[0], q_diff_range[1])
        # naming the x axis
        plt.xlabel('Timestep')
        # naming the y axis
        plt.ylabel('differences of qs')
        plt.legend()

    plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_metrics_' + method + '.pdf', dpi=200)
    print("Metrics of " + method + " plotted and saved successfully!!")


def plot_metrics(result_folder_dir, scenario_name,mstart,mstep,x,cum_violation, cum_action,r,qmax,cum_EC, q_diff, qt, existent_methods):
    print("Plotting metrics...")
    #fig = plt.figure(figsize=(35,16))
    fig, axs = plt.subplots(1, 7, figsize=(20, 30))
    plt.subplots_adjust(hspace=0.3)
    #qmax = 0.25 * (q_max + 1) - 1.5
    min_r_methods = []
    max_r_methods = []
    min_qmax_methods = []
    max_qmax_methods = []
    min_violation_methods = []
    max_violation_methods = []
    min_action_methods = []
    max_action_methods = []
    min_EC_methods = []
    max_EC_methods = []
    min_q_diff_methods = []
    max_q_diff_methods = []
    min_qt_methods = []
    max_qt_methods = []

    for idx, method in enumerate(existent_methods):
        transparency = 1 - 0.2 * idx
        plt.subplot(7, 1, 1)
        conv_r_method=np.convolve(np.array(r[idx][mstart:mstep+mstart]),np.ones(100)/100.,"same")
        min_r_methods.append(min(conv_r_method))
        max_r_methods.append(max(conv_r_method))
        plt.plot(x[mstart:mstep+mstart], conv_r_method, label=method, alpha=transparency)
        plt.legend()
        plt.xlabel('Timestep')
        plt.ylabel('reward')

        plt.subplot(7, 1, 2)
        conv_qmax_method = np.convolve(np.array(qmax[idx][mstart:mstep+mstart]),np.ones(1)/1., "same")
        min_qmax_methods.append(min(conv_qmax_method))
        max_qmax_methods.append(max(conv_qmax_method))
        plt.plot(x[mstart:mstep+mstart],conv_qmax_method, label=method, alpha=transparency)
        plt.legend()
        plt.xlabel('Timestep')
        plt.ylabel('Q_max')

        plt.subplot(7, 1, 3)
        conv_violation_method = np.convolve(np.array(cum_violation[idx][mstart:mstep+mstart]),np.ones(1)/1., "same")
        #conv_violation_method = cum_violation
        min_violation_methods.append(min(conv_violation_method))
        max_violation_methods.append(max(conv_violation_method))
        plt.plot(x[mstart:mstep+mstart-1],conv_violation_method[mstart:mstep+mstart-1], label=method, alpha=transparency)
        plt.ylabel('Moving average Violation (%)')
        plt.xlabel('Timestep')
        plt.legend()

        plt.subplot(7, 1, 4)
        min_action_methods.append(min(cum_action[idx][mstart:mstep+mstart]))
        max_action_methods.append(max(cum_action[idx][mstart:mstep+mstart]))
        plt.plot(x[mstart:mstep+mstart-1],cum_action[idx][mstart:mstep+mstart-1], label=method, alpha=transparency)
        plt.xlabel('Timestep')
        plt.ylabel('Moving average number of actions')
        plt.legend()

        plt.subplot(7, 1, 5)
        plt.plot(x[mstart:mstep + mstart-1], cum_EC[idx][mstart:mstep + mstart-1], label=method, alpha=transparency)
        min_EC_methods.append(min(cum_EC[idx][mstart:mstep + mstart]))
        max_EC_methods.append(max(cum_EC[idx][mstart:mstep + mstart]))
        # naming the x axis
        plt.xlabel('Timestep')
        plt.ylabel('Moving average energy consumption')
        # naming the x axis
        plt.legend()

        plt.subplot(7, 1, 6)
        #conv_qt_method = np.convolve(np.array(qt[idx][mstart:mstep + mstart]), np.ones(1) / 1., "same")
        #min_qt_methods.append(min(conv_qt_method))
        #max_qt_methods.append(max(conv_qt_method))
        min_qt_methods.append(min(qt[idx]))
        max_qt_methods.append(max(qt[idx]))
        plt.plot(qt[idx], label=method, alpha=transparency)
        plt.legend()
        plt.xlabel('Timestep')
        plt.ylabel('q')

        if "Double" in method or "DQN" in method:
            plt.subplot(7, 1, 7)
            min_q_diff_methods.append(min(q_diff[idx][mstart:mstep + mstart]))
            max_q_diff_methods.append(max(q_diff[idx][mstart:mstep + mstart]))
            plt.plot(x[mstart:mstep + mstart], q_diff[idx][mstart:mstep + mstart], label=method, alpha=transparency)
            plt.legend()
            plt.xlabel('Timestep')
            plt.ylabel('diff_q')

    # function to show the plot
    #plt.show()
    plt.savefig(result_folder_dir + '/plots_'+scenario_name+'_metrics.pdf', dpi=200)
    Min_reward = min(min_r_methods)
    Max_reward = max(max_r_methods)
    Min_qmax = min(min_qmax_methods)
    Max_qmax = max(max_qmax_methods)
    Min_cum_violation = min(min_violation_methods)
    Max_cum_violation = max(max_violation_methods)
    Min_cum_action = min(min_action_methods)
    Max_cum_action = max(max_action_methods)
    Min_cum_EC = min(min_EC_methods)
    Max_cum_EC = max(max_EC_methods)
    if len(min_q_diff_methods) > 0:
        Min_q_diff = min(min_q_diff_methods)
        Max_q_diff = max(max_q_diff_methods)
    else:
        Min_q_diff = 0
        Max_q_diff = 0
    Min_qt = min(min_qt_methods)
    Max_qt = max(min_qt_methods)
    print("Metrics plotted and saved successfully!!")

    for idx, method in enumerate(existent_methods):
        plot_metrics_seprately(result_folder_dir, scenario_name, mstart, mstep, x, cum_violation[idx], cum_action[idx], r[idx], qmax[idx],
                               cum_EC[idx], q_diff[idx], qt[idx], method, [Min_reward, Max_reward]
                               , [Min_qmax, Max_qmax], [Min_cum_violation, Max_cum_violation],
                               [Min_cum_action, Max_cum_action], [Min_cum_EC, Max_cum_EC], [Min_q_diff, Max_q_diff], [Min_qt, Max_qt])
        plot_metrics_seprately_without_range(result_folder_dir, scenario_name, mstart, mstep, x, cum_violation[idx], cum_action[idx],
                                             r[idx], qmax[idx], cum_EC[idx], q_diff[idx], qt[idx], method)

def plot_intervals_separately(result_folder_dir, scenario_name, x, lt, t_max, method, plot_name):
    print("Plotting violation for some intervals")
    mstart1 = mstart_list[0]
    mstart2 = mstart_list[1]
    mstart3 = mstart_list[2]
    mstart4 = mstart_list[3]
    mstep = mstep_interval
    if method == "SARSA":
        color = "orange"
    elif method =="Q":
        color ="blue"
    else:
        color = "green"
    fig = plt.figure(figsize=(16, 16))
    plt.subplot(4, 1, 1)
    plt.plot(x[mstart1:mstep + mstart1], lt[mstart1:mstep + mstart1], label=method, color=color)
    if len(t_max) > 0:
        plt.plot(x[mstart1:mstep+mstart1],t_max[mstart1:mstep+mstart1], label="Max_latency")
    #plt.ylim(latency_range[0], latency_range[1])
        # naming the x axis
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel(plot_name)
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(x[mstart2:mstep + mstart2], lt[mstart2:mstep + mstart2], label=method, color=color)
    if len(t_max) > 0:
        plt.plot(x[mstart2:mstep+mstart2],t_max[mstart2:mstep+mstart2], label="Max_latency")
    #plt.ylim(latency_range[0], latency_range[1])
    # naming the x axis
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel(plot_name)
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(x[mstart3:mstep + mstart3], lt[mstart3:mstep + mstart3], label=method, color=color)
    if len(t_max) > 0:
        plt.plot(x[mstart3:mstep+mstart3],t_max[mstart3:mstep+mstart3], label="Max_latency")
    #plt.ylim(latency_range[0], latency_range[1])
    # naming the x axis
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel(plot_name)

    # giving a title to my graph
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(x[mstart4:mstep + mstart4], lt[mstart4:mstep + mstart4], label=method, color=color)
    if len(t_max) > 0:
        plt.plot(x[mstart4:mstep+mstart4],t_max[mstart4:mstep+mstart4], label="Max_latency")
    #plt.ylim(latency_range[0], latency_range[1])
    plt.xlabel('Timestep')
    # naming the y axis
    plt.ylabel(plot_name)

    # giving a title to my graph
    plt.legend()
    # plt.savefig('workload_trace_taxi_{}'.format(x),format='pdf',dpi=600)
    # function to show the plot
    # plt.show()

    plt.savefig(result_folder_dir+'/plots_' + scenario_name + '_' + plot_name + '_' + method + '.pdf', dpi=200)
    print("Violation of " + method + " for some intervals plotted and saved successfully")


def plot_for_intervals(result_folder_dir, scenario_name,x,lt,t_max,existent_methods, plot_name):
    print("Plotting violation for some intervals")
    mstart1 = mstart_list[0]
    mstart2 = mstart_list[1]
    mstart3 = mstart_list[2]
    mstart4 = mstart_list[3]
    mstep = mstep_interval
    min_lt_methods = []
    max_lt_methods = []
    fig = plt.figure(figsize=(16, 16))
    t_max_shown = False
    for idx, method in enumerate(existent_methods):
        transparency = 1 - 0.2*idx
        min_lt_methods.append(min(lt[idx][mstart:mstep + mstart]))
        max_lt_methods.append(max(lt[idx][mstart:mstep + mstart]))
        plt.subplot(4, 1, 1)
        plt.plot(x[mstart1:mstep+mstart1],lt[idx][mstart1:mstep+mstart1], label=method, alpha=transparency)
        if len(t_max) > 0 and not t_max_shown:
            plt.plot(x[mstart1:mstep+mstart1],t_max[mstart1:mstep+mstart1], label="Max_latency")

        # naming the x axis

        plt.xlabel('Timestep')
        # naming the y axis
        plt.ylabel(plot_name)
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(x[mstart2:mstep+mstart2],lt[idx][mstart2:mstep+mstart2], label=method, alpha=transparency)
        if len(t_max) > 0 and not t_max_shown:
            plt.plot(x[mstart2:mstep+mstart2],t_max[mstart2:mstep+mstart2], label="Max_latency")
        # naming the x axis
        plt.xlabel('Timestep')
        # naming the y axis
        plt.ylabel(plot_name)
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(x[mstart3:mstep+mstart3],lt[idx][mstart3:mstep+mstart3], label=method, alpha=transparency)
        if len(t_max) > 0 and not t_max_shown:
            plt.plot(x[mstart3:mstep+mstart3],t_max[mstart3:mstep+mstart3], label="Max_latency")
        # naming the x axis
        plt.xlabel('Timestep')
        # naming the y axis
        plt.ylabel(plot_name)
        # giving a title to my graph
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(x[mstart4:mstep+mstart4], lt[idx][mstart4:mstep+mstart4], label=method, alpha=transparency)
        if len(t_max) > 0 and not t_max_shown:
            plt.plot(x[mstart4:mstep+mstart4], t_max[mstart4:mstep+mstart4], label="Max_latency")
            t_max_shown = True
        # naming the x axis
        plt.xlabel('Timestep')
        # naming the y axis
        plt.ylabel(plot_name)

        # giving a title to my graph
        plt.legend()
    #plt.savefig('workload_trace_taxi_{}'.format(x),format='pdf',dpi=600)
    # function to show the plot
    #plt.show()

    plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_' + plot_name + '.pdf', dpi=200)
    print("Violation for some intervals plotted and saved successfully")
    #Min_latency = min(min_lt_methods)
    #Max_latency = max(max_lt_methods)
    for idx, method in enumerate(existent_methods):
        plot_intervals_separately(result_folder_dir, scenario_name, x, lt[idx], t_max, method, plot_name)

def action_distribution(result_folder_dir, scenario_name,conf_distribution, existent_methods, setups):
    Color ={0: "r", 1:"b", 2: "g", 3:"c", 4:"m" }
    for method_idx, method in enumerate(existent_methods):
        ############## configs distribution among all steps ###############
        fig = plt.figure(figsize=(16, 10))
        my_counter = Counter(conf_distribution[method_idx])
        all_config = len(my_counter)
        x = []
        y = []
        for i in range(all_config):
            x.append(i + 1)
            y.append(my_counter[i + 1])

        fig = plt.figure(figsize=(16, 10))
        # plt.rcParams.update({'font.size': 18})
        # creating the bar plot
        plt.xticks(x, rotation=0)
        bars = plt.bar(x, y, color='maroon',
                       width=0.4)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x(), yval + .1, str(round(yval * 100 / max_step, 1)))

        plt.xlabel("Configuration")
        plt.ylabel("Number of times")
        plt.title("Distribution of configurations (%)")

        plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_all_distribution_config_' + method + '.pdf', dpi=300)

        ############## configs distribution for each interval ###############


        fig = plt.figure(figsize=(30, 10))
        setup = setups[method_idx]
        if "Var_Tmax" in setup["name_scenario"].keys():
            variation_list = setup["name_scenario"]["Var_Tmax"]
        elif "Var_Weight" in setup["name_scenario"].keys():
            variation_list = setup["name_scenario"]["Var_Weight"]
        else:
            print("Error: name_scenario must be 'Var_Tmax' or 'Var_Weight'")
            sys.exit(1)
        inter = max_step // len(variation_list)
        list_interval = list(range(1, max_step, inter))
        start_interval_idx = 0
        lable_list = []
        for idx, interval in enumerate(list_interval):
            x = []
            y = []
            lable_list.append("interval"+str(idx+1))
            if idx != len(list_interval)-1:
                my_counter_interval = Counter(conf_distribution[method_idx][interval-1:list_interval[idx+1]-1])
            else:
                my_counter_interval = Counter(conf_distribution[method_idx][interval - 1:max_step])
            for i in range(all_config):
                x.append(i+1)
                y.append(my_counter_interval[i+1])
            start_interval_idx = interval
            plt.xticks(x, rotation=0)
            bars = plt.bar([n+(idx*0.2) for n in x], y, color=Color[idx],
                           width=0.2, label=lable_list[idx])
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x(), yval + .1, str(round(yval * 100 / inter, 1)) )
        plt.margins(x=0.005)
        plt.xlabel("Configurations")
        plt.ylabel("Number of times")
        plt.title("Distribution of configurations (%)")
        plt.legend()
        plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_distribution_config_each_interval_' + method + '.pdf', dpi=300)
def action_distribution_large_configs(result_folder_dir, scenario_name,conf_distribution, existent_methods, setups,common_configs_num ):
    Color ={0: "r", 1:"b", 2: "g", 3:"c", 4:"m" }
    for method_idx, method in enumerate(existent_methods):
        ############## configs distribution among all steps ###############
        fig = plt.figure(figsize=(16, 10))
        my_counter = Counter(conf_distribution[method_idx])
        all_config = len(my_counter)
        x = []
        x_tick = []
        y = []

        my_top_counter = my_counter.most_common(common_configs_num)
        i = 0
        for counter in my_top_counter:
            x.append(i)
            x_tick.append(counter[0])
            y.append(counter[1])
            i += 1

        fig = plt.figure(figsize=(16, 10))
        # plt.rcParams.update({'font.size': 18})
        # creating the bar plot
        plt.xticks(ticks=x,labels=x_tick)
        bars = plt.bar(x, y, color='maroon',
                       width=0.4)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x(), yval + .1, str(round(yval * 100 / max_step, 1)))

        plt.xlabel("Configuration")
        plt.ylabel("Number of times")
        plt.title("Distribution of configurations (%)")

        plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_all_distribution_config_' + method + '.pdf', dpi=300)

        ############## configs distribution for each interval ###############


        fig = plt.figure(figsize=(50, 10))
        setup = setups[method_idx]
        if "Var_Tmax" in setup["name_scenario"].keys():
            variation_list = setup["name_scenario"]["Var_Tmax"]
        elif "Var_Weight" in setup["name_scenario"].keys():
            variation_list = setup["name_scenario"]["Var_Weight"]
        else:
            print("Error: name_scenario must be 'Var_Tmax' or 'Var_Weight'")
            sys.exit(1)
        inter = max_step // len(variation_list)
        list_interval = list(range(1, max_step, inter))
        start_interval_idx = 0
        lable_list = []
        x_tick = []
        x_total = []
        for idx, interval in enumerate(list_interval):

            lable_list.append("interval"+str(idx+1))
            if idx != len(list_interval)-1:
                my_counter_interval = Counter(conf_distribution[method_idx][interval-1:list_interval[idx+1]-1])
            else:
                my_counter_interval = Counter(conf_distribution[method_idx][interval - 1:max_step])

            x = []
            y = []

            my_top_counter = my_counter_interval.most_common(common_configs_num)
            i = 0
            for counter in my_top_counter:
                x.append(i+(common_configs_num*idx))
                x_total.append(i+(common_configs_num*idx))
                x_tick.append(counter[0])
                y.append(counter[1])
                i += 1

            start_interval_idx = interval

            bars = plt.bar(x, y, color=Color[idx],
                           width=0.3, label=lable_list[idx])
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x(), yval + .1, str(round(yval * 100 / inter, 1)))
        plt.margins(x=0.005)
        plt.xticks(ticks=x_total, labels=x_tick)
        plt.xlabel("Configurations")
        plt.ylabel("Number of times")
        plt.title("Distribution of configurations (%)")
        plt.legend()
        plt.savefig(result_folder_dir + '/plots_' + scenario_name + '_distribution_config_each_interval_' + method + '.pdf', dpi=300)
def extract_data_and_plot(data,all_setups,window_violation,max_step,mstart,mstep,r,qmax,path, losses, qt, existent_methods):
    scenario_name = getScenarioName(path)
    result_folder = ""
    for method in existent_methods:
        result_folder = method + "_" + result_folder
    result_folder_dir = os.getcwd() + "/" + result_folder
    if not os.path.exists(result_folder_dir):
        os.mkdir(result_folder_dir)

    if len(losses[0]) > 0:
        plot_loss(result_folder_dir, scenario_name, losses, existent_methods)
    print("Extracting data and plotting...")
    t_max=[]
    cum_action = []
    latencies = []
    mean_cum_violations = []
    lt = []
    cum_violation = []
    cum_EC = []
    action = []
    EC = []
    q_diff = []
    conf_distribution = []
    lt_shrinked_one_instance = []
    energy_cons_one_instance = []
    setups = []
   # pdb.set_trace()
    number_steps =[max_step]
    for method_idx in range(len(existent_methods)):
        number_steps.append(data[method_idx][0].shape[1])
    max_step = min(number_steps)
    mstep = min(number_steps)
    for method_idx in range(len(existent_methods)):
        rate_5G = []
        rate_WIFI = []
        l_cloud = []
        t_max = []
        action_method = []
        conf_distribution_method = []
        x = []
        previous_config = 0
        setups.append(all_setups[method_idx][0])
        for i in range(1, max_step+1):

            x.append(i)
            dt_method = data[method_idx][0].columns[i]

            data_json = json.loads('{"info":'+dt_method+'}')

            rate_5G.append(data_json["info"]["5G"])
            rate_WIFI.append(data_json["info"]["wifi"])
            l_cloud.append(data_json["info"]["l_cloud"])
            t_max.append(data_json["info"]["others"]["T_max"])
            act = data_json["info"]["action"]
            keys = list(act.keys())
            key = keys[0]
            act_v = act[key]
            if "config" in keys:
                action_method.append(int(act_v))#act[key]["config"]
                conf_distribution_method.append(int(act_v))
                previous_config = int(act_v)
            else:
                action_method.append(0)
                conf_distribution_method.append(previous_config)
        max_time_to_shrink = max(t_max) + 5
        cum_action_method = [getCumulativeAction(window_violation, t, mstep, action_method) for t in range(mstep)]
        cum_action.append(cum_action_method)
        action.append(action_method)
        conf_distribution.append(conf_distribution_method)
        latencies_method = []
        cum_violation_method=[]
        for t in range(0, len(data[method_idx])):
            # data[t].shape[1]
            latency_exp =[]
            for j in range(1, max_step+1):
                df = data[method_idx][t].columns[j]
                _data_json = json.loads('{"info":' + df + '}')
                latency_exp.append(_data_json["info"]["others"]["latency"])
            cum_violation_exp=[getViolationInterval1(window_violation, t + 1, max_step, t_max, latency_exp) for t in range(mstep)]
            latencies_method.append(latency_exp)
            cum_violation_method.append(cum_violation_exp)
        mean_cum_violation_method = np.mean(np.asarray(cum_violation_method), 0)
        mean_cum_violations.append(mean_cum_violation_method)
        latencies.append(latencies_method)
        lt_method_shrinked_one_instance = [shrink(value, max_time_to_shrink) for value in latencies_method[0]]
        lt_shrinked_one_instance.append(lt_method_shrinked_one_instance)
        print("Getting moving average violation...")
        lt_method_shrinked = [[shrink(value, max_time_to_shrink) for value in latencies_method[idx_exp]] for idx_exp in range(len(latencies_method))]
        lt.append(lt_method_shrinked)
        #cum_violation.append([getViolationInterval1(window_violation, t + 1, max_step, t_max, lt_method_shrinked) for t in range(mstep)])

        energy_cons_method = []
        for t in range(0, len(data[method_idx])):
            # data[t].shape[1]
            energy_cons_exp = []
            for j in range(1, max_step+1):
                df = data[method_idx][t].columns[j]
                _data_json = json.loads('{"info":' + df + '}')
                energy_cons_exp.append(_data_json["info"]["others"]["energy"])
            energy_cons_method.append(energy_cons_exp)
        energy_cons_one_instance.append(energy_cons_method[0])
        energy_cons_mean = np.mean(np.asarray(energy_cons_method), 0)
        EC.append(energy_cons_mean)
        print("Getting moving average energy consumption...")
        cum_EC_method = [getEnergyConsumptionInterval1(window_violation, t, mstep, energy_cons_mean) for t in range(mstep)]
        cum_EC.append(cum_EC_method)

        q_diff_method = []
        for t in range(0, len(data[method_idx])):
            # data[t].shape[1]
            q_diff_exp = []
            for j in range(1, max_step + 1):
                df = data[method_idx][t].columns[j]
                _data_json = json.loads('{"info":' + df + '}')
                q_diff_exp.append(_data_json["info"]["others"]["max_diff_Qs"])
            q_diff_method.append(q_diff_exp)
        q_diff_mean = np.mean(np.asarray(q_diff_method), 0)
        q_diff.append(list(q_diff_mean))
    my_counter = Counter(conf_distribution[0])
    all_config = len(my_counter)
    if all_config>common_configs_numbers:
        action_distribution_large_configs(result_folder_dir, scenario_name,conf_distribution,existent_methods, setups,common_configs_numbers)
    else:
        action_distribution(result_folder_dir, scenario_name,conf_distribution,existent_methods, setups)
    plot_metrics(result_folder_dir, scenario_name,mstart,mstep,x,mean_cum_violations, cum_action,r,qmax, cum_EC, q_diff, qt, existent_methods)
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.size': 14})
    plot_for_intervals(result_folder_dir, scenario_name,x,lt_shrinked_one_instance,t_max, existent_methods, 'Latency(ms)')
    plot_for_intervals(result_folder_dir, scenario_name, x, conf_distribution, [], existent_methods, 'Actions\' number')
    plot_for_intervals(result_folder_dir, scenario_name, x, energy_cons_one_instance, [], existent_methods, 'Normalized energy consumption')
    interval = window_violation
    countViolation(result_folder_dir, scenario_name, interval, mstep, t_max, lt, existent_methods)



    print("Data extraction and plotting done successfully")
        

def make_all_plots(dir,n_experiments,window_violation,maxstep,mstart,mstep):
    print("Making all the plots")

    for var_folder in os.listdir(dir):
        if os.path.isdir(os.path.join(dir,var_folder)):
            Path1 = os.path.join(dir,var_folder)
            for lr_folder in os.listdir(Path1):
                if os.path.isdir(os.path.join(Path1, lr_folder)):
                    Path2 = os.path.join(Path1, lr_folder)
                    for Var in os.listdir(Path2):
                        if os.path.isdir(os.path.join(Path2, Var)):
                            Path3 = os.path.join(Path2, Var)
                            r = []
                            qmax = []
                            qt = []
                            all_data_files = []
                            all_setups = []
                            core_params = get_core_param(n_experiments, methods, Path3)

                            out = Parallel(n_jobs=5)(
                                delayed(load_data)(core_param, Path3) for core_param in core_params)
                            outs = {}
                            for i in out:
                                *key, _ = i[-1]
                                outs.setdefault(tuple(key), []).append(i)
                            all_outs = list(outs.values())

                            losses = []
                            existent_methods = []
                            for data_method in all_outs:
                                existent_methods.append(data_method[0][3])
                                data_file = []
                                setup = []
                                loss = []
                                for idx, data in enumerate(data_method):
                                    method = data[3]
                                    data_file.append(data[0])
                                    setup.append(data[2])
                                    method_Path = os.path.join(Path3, method)
                                    if "DQN" in method:
                                        loss.append(data[1])
                                all_data_files.append(data_file)
                                all_setups.append(setup)
                                losses.append(loss)
                                r.append(np.load(os.path.join(method_Path, "original_r.npy")))
                                qmax.append(np.load(os.path.join(method_Path,'maxQ.npy')))
                                qt.append(np.load(os.path.join(method_Path,'Qtmean.npy')))
                            extract_data_and_plot(all_data_files, all_setups,window_violation,maxstep,mstart,mstep,r,qmax,Path3, losses,qt, existent_methods)
    #print("All the plots made and save successfully",file=file)


methods = ["QLearning"]
#methods = ["DQN"]
directory = "app8/logs/12_11_2023_18_29_33" #"/Users/hamtasedghani/PMAIEDGE/PMAIEDGE/Applications/logs/14_11_2023_10_50_26"
n_experiments = 10
window_violation=1000
max_step=1500000
mstart=0
mstep=1500000
common_configs_numbers=15
mstep_interval=2000
mstart_list = [0,(max_step//3)-mstep_interval, (2*max_step//3)-mstep_interval, max_step-mstep_interval]
n_steps_per_fit = 5
initial_replay_size = 1000
is_completed = True

if not is_completed:
    legend_labels = []
    fig = plt.figure(figsize=(16, 8))
    output = []
    new_directory = os.path.join(directory, "completed_up_to_checkpoint")
    '''if not os.path.exists(new_directory):
        os.mkdir(new_directory)
    else:
        os.remove(new_directory)
        os.mkdir(new_directory)
    for file_name in os.listdir(directory):
        if file_name != "completed_up_to_checkpoint":
            source = os.path.join(directory, file_name)
            if os.path.isdir(source):
                dest = os.path.join(new_directory, file_name)
                shutil.copytree(source, dest)
            else:
                shutil.copy(source, new_directory)'''
    abs_path = os.path.abspath(new_directory)
    for path, directories, files in os.walk(abs_path):
        for Dir in directories:
            if "exp_" in Dir:
                test = os.listdir(os.path.join(path, Dir))
                for item in test:
                    if item.endswith(".zip"):
                        s = [float(s) for s in re.findall(r'-?\d+\.?\d*', item)]
                        if len(s) > 0:
                            current_step = int(s[0])
                            break
                        else:
                            print("There is no trustworthy checkpoints")
                            sys.exit()
                for item in test:
                    if item.startswith("training_"):
                        os.rename(os.path.join(os.path.join(path, Dir), item),
                                  os.path.join(os.path.join(path, Dir), "training_{}.txt".format(current_step)))
                        break
                setup_json = os.path.join(path, Dir) + "/setup.json"
                with open(setup_json) as f:
                    setup = json.load(f)
                setup["n_steps"] = current_step
                system = json.dumps(setup, indent=2)
                with open(setup_json, "w") as f:
                    f.write(system)
                output.append(experiment_completed(os.path.join(os.path.join(path, Dir))))
    outs = {}
    for i in output:
        *key, _ = i[-1]
        outs.setdefault(tuple(key), []).append(i)
    #import pdb
    #pdb.set_trace()
    all_outs = list(outs.values())
    for out in all_outs:
        equal_length = all(len(ele[0]) == len(out[0][0]) for ele in out)
        if not equal_length:
            length_r = min([len(out[i][0]) for i in range(len(out))])
            for id, loss in enumerate(out):
                out_list = list(out[id])
                out_list[0] = loss[0][0:length_r]
                out[id] = tuple(out_list)
        equal_length = all(len(ele[1]) == len(out[0][1]) for ele in out)
        if not equal_length:
            length_maxq = min([len(out[i][1]) for i in range(len(out))])
            for id, loss in enumerate(out):
                out_list = list(out[id])
                out_list[1] = loss[1][0:length_maxq]
                out[id] = tuple(out_list)
        r = np.array([o[0] for o in out])
        max_Qs = np.array([o[1] for o in out])
        n_steps = np.array([o[2] for o in out])
        q = np.array([o[3] for o in out])
        # qt = np.array([o[4] for o in out])
        # directory4 = out[0][5][0]
        directory4 = out[0][4][0]

        np.save(directory4 + '/original_r.npy', np.mean(r, 0))
        r = np.convolve(np.mean(r, 0), np.ones(100) / 100., 'valid')
        # q_t = np.mean(qt, 0)
        # qt_s = np.convolve(q_t, np.ones(100) / 100., 'valid')

        max_Qs = np.mean(max_Qs, 0)

        plt.subplot(3, 1, 1)
        plt.plot(r)
        plt.legend(["reward"])
        plt.subplot(3, 1, 2)
        plt.plot(max_Qs, color='magenta')
        legend_labels.append("Qmax")
        # plt.subplot(3, 1, 3)
        # plt.plot(qt_s, color='green')
        # legend_labels.append("q")

        plt.legend(legend_labels)
        fig.savefig(directory4 + "/" + 'test.png')
        # print("directory"+mydir)
        np.save(directory4 + '/r.npy', r)
        np.save(directory4 + '/maxQ.npy', max_Qs)
        np.save(directory4 + '/Q.npy', q)
    directory = new_directory
make_all_plots(directory,n_experiments, window_violation,max_step,mstart,mstep)
