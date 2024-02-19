

from abc import ABC, abstractmethod


class Action(ABC):
    """This class represents a generic action

    Args:
        id (str):  Unique keyword that identifies the action, it is used as key in the action factory
        
        description (str): Action description
        
        **kwargs :  Keyword arguments
        
        
    """    

    def __init__(self, action_id, description, **kwargs):
     
        self.id = action_id
        self.description = description

    @abstractmethod
    def apply(self, configuration):
        """apply a action on a given configuration

        Args:
            configuration (AIapp.Configuration): configuration to which he action will be applied
        """        

        pass

    @abstractmethod
    def check_feasibility(self, configuration, system):
        """method to check the feasibility of an action in a given configuration

        Args:
            configuration (AIapp.Configuration): The current configuration of the system
            system (System.System): the system
        """        

        pass

    def __str__(self):
        """convert the action to string to log it

        Returns:
            str : return the string that has the action description
        """        
        s = '"action":{'
        s += '"name":"{}"'.format(self.id)
        s += ","
        s += '"description": "{}"'.format(self.description)
        s += '}'
        return s


class DoNothing(Action):
    """Class that define the do nothing action. If this action is choosen then the system configuration remains the same

    Args:
        id (str):  Unique keyword that identifies the action
    """    

    def __init__(self, action_id, **kwargs):

        super().__init__(action_id, "Do nothing")

    def apply(self, configuration):
        """apply the DoNothing action on the given configuration

        Args:
            configuration (AIapp.Configuration): configuration to which he action will be applied
        """ 

        return

    def check_feasibility(self, configuration, system):
        """return True since the DoNothing action is possible in all configurations

        Args:
            configuration (AIapp.Configuration): The current configuration of the system
            system (System.System): The system

        Returns:
            bool: return True since the DoNothing action is always feasible
        """
        return True


class ChangeConfiguration(Action):
    """Define the actions that change the current configuration

    Args:
        action_id (str): the uniue keyword that identifies the action
        config (AIapp.Configuration): the system configuration that is associated to the action
    """
    def __init__(self, action_id, config, **kwargs):
       
        super().__init__(action_id, "Change the DNN configuration of the system")
        self.config = config

    def apply(self, configurations):
        """return the system configuration that is choosen by the action

        Args:
            configurations (list(AIapp.Configuration)): list of all system configurations

        Returns:
            AIapp.Configuration: the configuration choosen by the action
        """
        return configurations[self.id-1]

    def check_feasibility(self, configuration, system):
        """check if the action is feasible in the current configuration

        Args:
            configuration (AIapp.Configuration): the current configuration
            system (System.System): the system

        Returns:
            bool: return True if the choosen action is feasible in the when the system is in the current configuration
        """
        return True

    def __str__(self):
        """convert the action in a string that show the action description

        Returns:
            str: String that has the action description
        """        
        s = super().__str__()
        s = s[:-1] + ', "config": "{}"'.format(self.config.name)
        s += '}'
        return s