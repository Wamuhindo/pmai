"""Copyright 2023 Abednego Wamuhindo Kambale

    """
from classes.Logger import Logger

import collections
import numpy as np
import sys
import json


class LatencyManager:
    """a class to represent the Latency manager

        Args:
            system (_type_): _description_
            file_config (_type_, optional): _description_. Defaults to None.
            json_config (_type_, optional): _description_. Defaults to None.
    """

    def __init__(self, system, file_config=None, json_config=None):
       
        self.logger = Logger(stream=system.logger.stream,
                             level=system.logger.level,
                             verbose=system.logger.verbose,)
        self.error = Logger(stream=sys.stderr, verbose=1, is_error=True)

        self.logger.log("Initializing latency manager...", 1)
        self.logger.level += 1
