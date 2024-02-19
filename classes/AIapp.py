"""Copyright 2023 Abednego Wamuhindo Kambale

    """


class AIapp:
    """A class to represent the AI application
        Args:
            name (str): the AI application name
            configurations (list of AIapp.Configuration): list of DNN configuartions of the DNN
    """

    def __init__(self, name, configurations):
        
        self.name = name
        self.configurations = configurations

    def __eq__(self, other):
        """function that returns true if self.name equals other.name

        Args:
            other (AIapp): the other AI application
        """
        return self.name == other.name

    def __str__(self):
        """convert the AI application into a string
        """
        s = '"' + self.name + '": {'
        for configuration in self.configurations:
            s += (configuration.__str__() + ',')
        s = s[:-1] + '}'
        return s


class Configuration:
    """A class to represent the Configuration
        Args:
            name (str): name of the DNN configuration
            data_to_phone (str): amount of data that can be send to the phone
            data_to_cloud (str):amount of data that can be send to the cloud
            partitions (list of Partitions): a list of patitions
    """

    def __init__(self, name, partitions, data_to_phone, data_to_cloud):

        self.name = name
        self.partitions = partitions
        self.data_to_phone = data_to_phone
        self.data_to_cloud = data_to_cloud

    def __eq__(self, other):
        """function that returns true if self.name equals other.name

        Args:
            other (AIapp.Configuration): the other configuration
        """
        return self.name == other.name

    def __str__(self):
        """convert the AI application into a string
        """
        s = '"{}":'.format(self.name)
        s += '{ "data_to_phone": '
        s += '{}, "data_to_cloud":{},'.format(
            self.data_to_phone, self.data_to_cloud)
        s += ' "partitions" :{ '
        for partition in self.partitions:
            s += (partition.__str__() + ',')
        s = s[:-1] + '}'
        s += "}"
        return s


class Partition:
    """ A class to represent the partition of the Configuration
        Args:
            name (str): the name of the partition,
            fake_name (str): a fake_name that is given to the partition when logging in a file
            memory (float): the memory occuped by the partition
            workload (float): the number of floating point operation of the partition
            latency (float): the latency of the partition
            data_out (float): the data sent to the next partition
    """

    def __init__(self, fake_name, name, memory, workload, latency, data_out):

        self.fake_name = fake_name
        self.name = name
        self.memory = memory
        self.workload = workload
        self.latency = latency
        self.data_out = data_out

    def __eq__(self, other):
        """function that returns true if self.name equals other.name

        Args:
            other (AIapp.Configuration): the other configuration
        """
        return self.name == other.name

    def __str__(self):
        """convert the partition into a string
        """
        s = '"{}": {{"name":"{}","memory":{}, "workload":{}, "latency":{}, "data_out":{}}}'.format(
            self.fake_name, self.name, self.memory, self.workload, self.latency, self.data_out)
        return s
