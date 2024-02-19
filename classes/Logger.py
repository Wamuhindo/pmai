
from datetime import datetime
import sys


class Logger:
    """_summary_

    Args:
        stream (_type_, optional): the stream where to send the logging. Defaults to sys.stdout.
        level (int, optional): the logging level. Defaults to 0.
        is_error (bool, optional): a boolean specifying if the logger is for error. Defaults to False.
        verbose (int, optional): the logger verbosity. Defaults to 0.
    """
    def __init__(self, stream=sys.stdout,  level=0, is_error=False, verbose=0):

        self.stream = stream
        self.verbose = verbose
        self.level = level
        self.is_error = is_error

    def init_log(self, msg):
        """method that initializes the logger

        Args:
            msg (str): the message to be streamed

        Returns:
            str: the message to be logged
        """        

        log_msg = " " * 4 * self.level
        if self.is_error:
            log_msg += "ERROR: "
        log_msg += msg
        return log_msg

    def log(self, msg, verb=0):
        """method that logs the message

        Args:
            msg (str): the message to be logged
            verb (int, optional): the verbosity level of the message. Defaults to 0.
        """        
        log_msg = ""
        if verb <= self.verbose:
            log_msg = self.init_log(msg)
            print(log_msg, file=self.stream)

    def __set_log_state__(self, dictionary):
        """Method to support pickling/unpickling of Logger objects

        Args:
            dictionary (dict): the dictionnary
        """        

        if "stream" in dictionary:
            stream = Logger.parse_wrapper(dictionary["stream"])
            dictionary["stream"] = stream
        self.__dict__.update(dictionary)

    def __get_log_state__(self):
        """Method to support pickling/unpickling of Logger objects

        Returns:
            dict: dictionary
        """        

        dictionary = self.__dict__.copy()
        if "stream" in dictionary:
            if type(dictionary["stream"]) == "_io.TextIOWrapper":
                stream_property_list = [
                    dictionary["stream"].name, dictionary["stream"].mode]
            else:
                stream_property_list = [dictionary["stream"].name]
            dictionary["stream"] = stream_property_list
        return dictionary

    def print_time(self, time):
        
        """method that print the logging time
        """        

        now = datetime.now()
        print("TIME#{} --> {}".format(now, time), file=self.stream)

    @staticmethod
    def parse_wrapper(wrapper):
        """ Method to convert a list of stream properties into an actual stream


        Args:
            wrapper: The list of stream properties. It is defined as [file_name, mode] if the stream is a _io.TextIOWrapper, while it stores [stdout] if the stream is sys.stoud ([sterr], respectively)

        Returns:
            Stream: A stream
        """
        if len(wrapper) > 1:
            name = wrapper[0]
            mode = wrapper[1]
            stream = open(name, mode)
        elif wrapper[0] == "stdout":
            stream = sys.stdout
        else:
            stream = sys.stderr
        return stream
