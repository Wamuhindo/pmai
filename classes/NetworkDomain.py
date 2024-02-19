"""Copyright 2023 Abednego Wamuhindo Kambale

    """
import numpy as np
import pandas as pd
import random
import copy
#random.seed(1234)


class NetworkDomain:
    """a class to represent a generic network Domain
      Args:
            name (str): the name of the network domain
            data_rate (float): the data_rate of the network domain
    """
    MAX_STEP = 19270
    #MAX_STEP = 3344

    def __init__(self, name,  data_rate,variations=[], trace=None, cost_per_byte=0,env_number=1,max_env_step=10000):

        self.name = name
        self.data_rate = data_rate
        self.cost_per_byte = cost_per_byte
        self.system_variation = variations
        self.env_number = env_number
        self.max_env_step = max_env_step
        if trace is not None:
            self.trace = trace
        else:
            self.trace = ""
        self.data_rates = self.init_data_rates()

    def __eq__(self, other):
        """function that returns true if self.name equals other.name

        Args:
            other (NetworkDomain): the other NetworkDomain
        """
        return self.name == other.name

    def __str__(self):
        """convert the Network Domain into a string
        """
        s = '"{}": '.format(self.name)
        s += "{"
        s += '"data_rate":{},"trace":"{}"'.format(
            self.data_rate, self.trace)
        s += "}"
        return s

    def _get_rate(self, time_step):
        """return the throughput of the network domain 

        Args:
            time_step (int): the time step that is considered

        Returns:
            float: the throughput at the timestep
        """  
        if self.trace != "" and self.trace is not None:
            if self.name == "WIFI7":
                if "wifi" in self.system_variation:
                    return self.data_rates[time_step-1]
                else:
                    return self.data_rate
            else:
                if "5G" in self.system_variation:
                    return self.data_rates[time_step-1]
                else:
                    return self.data_rate
        else:
            return self.data_rate

    def init_data_rates(self):
        """method that initializes the data rate from the trace

        Args:
            trace (str): the trace file

        Returns:
            list: the data rate for different timesteps
        """        
        data_rates = []

        if self.name == "WIFI7":
            if self.trace != "" and self.trace is not None:
                b_trace = pd.read_csv(self.trace, delimiter=',')
                '''data_rates = [b_trace.iloc[i]["Throughput"] * 35
                              for i in range(0, b_trace.shape[0])]'''
                if "validation" in self.trace:
                    data_rates = [b_trace.iloc[i]["Throughput"]
                                for i in range(0, b_trace.shape[0])]
                else:
                    data_rates = [b_trace.iloc[i]["Throughput"] * 8
                                for i in range(0, b_trace.shape[0])]

            #return np.clip(data_rates,0.5,self.data_rate)
        else:
            if self.trace != "" and self.trace is not None:
                b_trace = pd.read_csv(self.trace, delimiter=',')
                data_rates = [b_trace.iloc[i]["Throughput"]
                              for i in range(0, b_trace.shape[0])]
        
        #uncomment the following line for validation
        len_list = self.max_env_step - len(data_rates)
        #uncomment the following line for validation
        #len_list = self.max_env_step - len([])
        random_list = []
        if len_list > 0:
            random_list = self.generate_random_list(data_rates+data_rates[::-1],len_list)

        #uncomment the following line for validation
        data_rates = data_rates + random_list
        #uncomment the following line for validation
        #data_rates = random_list

        dr = np.clip(data_rates,0.5,self.data_rate)
        # dictionary of lists
        #dict_pd = {'Throughput': dr}
            
        #df = pd.DataFrame(dict_pd)

        #df.to_csv(f"traces/generated/{self.name}_env_{self.env_number}.csv",index=False)

        return dr
    
    def generate_random_list(self,base_list, size):
        random_list = []
        base_list_copy = copy.deepcopy(base_list)
        for _ in range(size):

            if len(base_list_copy) == 0:
                base_list_copy = copy.deepcopy(base_list)
            # Get a random number from the base list
            base_number = random.choice(base_list_copy)
            base_list_copy.remove(base_number)
            
            # Generate a random percentage between -10% and 10%
            percentage_change = random.uniform(-0.1, 0.1)
            
            # Calculate the new number with the percentage change
            new_number = base_number * (1 + percentage_change)
            
            # Append the new number to the random list
            random_list.append(new_number)
        
        return random_list
