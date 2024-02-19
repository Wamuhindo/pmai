"""Copyright 2023 Abednego Wamuhindo Kambale

    """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # plotting
import random

class Device:
    """ a class to represent a generic device
        Args:
                name (str): the device name
                power_compute (float): the power consumed when computing one floating-point operation
                power_compute (float): the power consumed by the device network interface while transmitting data
                battery_state (float): the current percentage of the battery
    """
    MAX_STEP = 19270

    def __init__(self, name, power_compute=0, power_transmission=0, battery_state=100, trace=None, variations=[]):

        self.name = name
        self.power_compute = power_compute
        self.power_transmission = power_transmission
        self.battery_state = battery_state
        self.system_variation = variations
        if trace is not None:
            self.trace = trace
        else:
            self.trace = ""
        self.battery = []
        self.battery = Device.init_battery(self, self.trace)

    def __eq__(self, other):
        """function that returns true if self.name equals other.name

        Args:
            other (Device.Device): the other Device
        """
        return self.name == other.name

    def __str__(self):
        """convert the Device into a string
        """
        s = '"{}": '.format(self.name)
        s += '{ '
        s += '"power_compute":{}, "power_transmission":{},"battery_state":{},"trace":"{}"'.format(
            self.power_compute, self.power_transmission, self.battery_state, self.trace)
        s += '}'
        return s

    def _get_battery(self, timestep):

        return self.battery[(timestep - 1) % Device.MAX_STEP]

    @staticmethod
    def create_element_bat(trace):
        """create a battery element

        Args:
            trace (str): the bttery trace to be considered

        Returns:
            dict {str:str}: a dictionnary that contains the value of the battery charge and the charging status
        """
        el = dict()
        el["value"] = trace["Percentage"]
        el["plugged"] = trace["Charging"]
        return el

    @staticmethod
    def init_battery(device, trace):
        """method that initializes the battery status

        Args:
            device (Device.Device): the device to consider
            trace (str): the battery trace (a csv file)

        Returns:
            list: battery list that contains the values of the battery percentage
        """
        battery = []
        if trace != "" and trace is not None:
            b_trace = pd.read_csv(trace, delimiter=',')

            battery = [Device.create_element_bat(
                b_trace.iloc[i]) for i in range(0, b_trace.shape[0])]
        else:
            battery = [Device.create_element_bat(dict(
                {"Percentage": device.battery_state, "Charging": False})) for i in range(0, 19270)]
        return battery


class CloudServer:
    """ a class to represent the cloud server
        Args:
            name (name): name of the cloud server
            workload (a function of the Workload.Workload class): the workload function
    """
    MAX_STEP = 3000000

    def __init__(self, name, latency_min=2, latency_max=6, trace=None, variations=[]):

        self.name = name
        self.latency_min = latency_min
        self.latency_max = latency_max
        self.system_variation = variations

        if trace is not None:
            self.trace = trace
            self.workloads = [0]
        else:
            self.trace = ""
            self.workloads = []
        '''self.workloads, self.latencies = CloudServer.init_workloads(
            self, self.trace)'''
        #self.latencies = CloudServer.init_latencies(self)

        mstart = 0
        mstep = 3000

        '''
        fig = plt.figure(figsize=(16, 16))
        plt.subplot(3, 1, 1)
        plt.plot(self.latencies[0][mstart:mstep + mstart])
        # naming the x axis
        plt.xlabel('Timestep')
        # naming the y axis
        plt.ylabel('Latency(ms)')
        plt.legend(["Cloud latency"])
        plt.subplot(3, 1, 2)
        plt.plot(self.latencies[1][mstart:mstep + mstart])
        # naming the x axis
        plt.xlabel('Timestep')
        # naming the y axis
        plt.ylabel('Latency(ms)')
        plt.legend(["Cloud latency"])
        plt.subplot(3, 1, 3)
        plt.plot(self.latencies[2][mstart:mstep + mstart])
        # naming the x axis
        plt.xlabel('Timestep')
        # naming the y axis
        plt.ylabel('Latency(ms)')

        # giving a title to my graph
        plt.legend(["Cloud Latency"])

        plt.savefig('trace.png', dpi=300)'''
    def __eq__(self, other):
        """function that returns true if self.name equals other.name

        Args:
            other (CloudServer): the other configuration
        """
        return self.name == other.name

    def __str__(self):
        """convert the Cloud server into a description string
        """
        s = '"{}": '.format(self.name)
        s += '{ '
        s += '"latency_min":{},"latency_max":{},"trace":"{}"'.format(
            self.latency_min, self.latency_max, self.trace)
        s += "}"
        return s

    def _get_latency(self, time_step, configuration):
        """Method that returns the cloud latency at a specific time-step

        Args:
            time_step (int): the time step at which the cloud latency is needed

        Returns:
            float: cloud latency
        """
        latency = configuration.partitions[2].latency
        if "cloud" not in self.system_variation or latency == 0:
            return latency
        else:
            return self.get_config_latency(latency)


    def _get_workload(self, time_step):
        """metho that returns the cloud workload at a given time_step

        Args:
            time_step (int): the time step

        Returns:
            float: cloud workload at the given timestep
        """
        if len(self.workloads) == 0:
            return 0
        else:
            # return self.workloads[(time_step-1) % CloudServer.MAX_STEP]
            return 0

    @staticmethod
    def get_workload(workload, extremum):
        """method to build the workload from a trace

        Args:
            workload (float): the workload
            extremum (dict:{str:float}): the extremums of the workload

        Returns:
            float: workload
        """
        if workload > extremum["max"]:
            extremum["max"] = workload
        if workload < extremum["min"]:
            extremum["min"] = workload
        return workload

    def get_config_latency(self, latency):
        size = 50
        variation = 2
        mean = latency  # Calculate the mean within the range
        lmbda = 1 / mean  # Calculate the lambda parameter
        values = np.random.exponential(scale=1 / lmbda, size=size) + (latency - variation)
        values = np.clip(values, latency - variation, latency + variation)
        return random.choice(values)
    @staticmethod
    def workload_to_latency(device, extremum, workload):
        """extract the latency from the workload

        Args:
            device (Device.Device): the device to consider. In this case the cloud device
            extremum (dict:{str:float}): the extremums of the workload
            workload (float): the workload

        Returns:
            latency: the latency that is computed from the workload
        """
        # implement the relation between the latency and the workload
        # Calculate the slope of the line
        slope = (device.latency_max - device.latency_min) / \
                (extremum["max"] - extremum["min"])

        # Calculate the y-intercept of the line
        y_intercept = device.latency_min - slope * extremum["min"]

        return slope * workload + y_intercept

    @staticmethod
    def init_workloads(device, trace):
        """Method that initializes the workloads

        Args:
            device (Device.Device): The cloud device
            trace (trace): the workload trace

        Returns:
            list,list: workload list and latency list
        """
        workloads = []
        latencies = []
        extremum = dict({"min": 0, "max": 0})
        if trace != "" and trace is not None:
            workloads = [0]
            '''b_trace = pd.read_csv(trace, delimiter=',')
            workloads = [CloudServer.get_workload(b_trace.iloc[i]["workload"], extremum)
                         for i in range(0, b_trace.shape[0])]
            latencies = [CloudServer.workload_to_latency(
                device, extremum, wkld) for wkld in workloads]'''
        return workloads, latencies

    @staticmethod
    def init_latencies2(device):
        """initialize the latency from two values limit using an exponential distribution

        Args:
            device (Device.Device): the cloud device

        Returns:
            list: the list of latencies at different time_steps
        """
        # set the lower and upper bounds of the desired range
        lat_min = [720, 800, 850]  # device.latency_min
        lat_max = [750, 840, 910]  # device.latency_max
        # mean = [735,820,900]

        # set the mean of the exponential distribution
        # mean = (lat_min + lat_max) / 2

        # set the maximum variation in value
        max_var = 2

        # set the number of values to generate
        size = CloudServer.MAX_STEP

        # initialize an empty list to store the generated values
        values = []
        values_set = []

        # take the mean of the exponential distributuion among 2,4,6 and generate the values
        for i in range(0, 3):
            # generate the first value
            # np.random.seed(5)
            mean = (lat_max[i] - lat_min[i]) / 2  # Calculate the mean within the range
            lmbda = 1 / mean  # Calculate the lambda parameter
            values = np.random.exponential(scale=1 / lmbda, size=size) + lat_min[i]
            values = np.clip(values, lat_min[i], lat_max[i])

            values_set.append(values)

        # print the resulting list
        return values_set

    @staticmethod
    def init_latencies(device):
        """initialize the latency from two values limit using an exponential distribution

        Args:
            device (Device.Device): the cloud device

        Returns:
            list: the list of latencies at different time_steps
        """
        # set the lower and upper bounds of the desired range
        lat_min = [1, 3, 5]  # device.latency_min
        lat_max = [3, 5, 7]  # device.latency_max
        # mean = [2,4,6]

        # set the mean of the exponential distribution
        # mean = (lat_min + lat_max) / 2

        # set the maximum variation in value
        max_var = 2

        # set the number of values to generate
        size = CloudServer.MAX_STEP

        # initialize an empty list to store the generated values
        values = []
        values_set = []

        # take the mean of the exponential distributuion among 2,4,6 and generate the values
        for i in range(0, 3):
            # generate the first value
            # np.random.seed(5)
            mean = (lat_max[i] - lat_min[i]) / 2  # Calculate the mean within the range
            lmbda = 1 / mean  # Calculate the lambda parameter
            values = np.random.exponential(scale=1 / lmbda, size=size) + lat_min[i]
            values = np.clip(values, lat_min[i], lat_max[i])

            values_set.append(values)

        # print the resulting list
        return values_set

    @staticmethod
    def init_latencief(device):
        """initialize the latency from two values limit using an exponential distribution

        Args:
            device (Device.Device): the cloud device

        Returns:
            list: the list of latencies at different time_steps
        """
        # set the lower and upper bounds of the desired range
        lat_min = device.latency_min
        lat_max = device.latency_max

        # set the mean of the exponential distribution
        mean = (lat_min + lat_max) / 2

        # set the maximum variation in value
        max_var = 3

        # set the number of values to generate
        m = CloudServer.MAX_STEP

        # initialize an empty list to store the generated values
        values = []
        values_set = []

        # take the mean of the exponential distributuion among 2,4,6 and generate the values
        for mean in [2, 4, 6]:
            # generate the first value
            # np.random.seed(5)
            value = np.random.exponential(scale=mean)
            while value < lat_min or value > lat_max:
                value = np.random.exponential(scale=mean)

            # add the first value to the list
            values.append(value)

            # generate the remaining values
            for i in range(1, m):
                # generate a new value within the desired range
                while True:
                    new_value = np.random.exponential(scale=mean)
                    if abs(new_value - value) <= max_var and lat_min <= new_value and new_value <= lat_max:
                        break
                # add the new value to the list
                values.append(new_value)
                # update the current value
                value = new_value

            values_set.append(values)

        # print the resulting list
        return values_set
