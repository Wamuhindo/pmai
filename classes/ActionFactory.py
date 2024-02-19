from classes.Action import DoNothing, ChangeConfiguration


class ActionFactory:
    """ Class to define a factory of actions, characterized by an id that identifies the action type
    
    Args:
        actions (dict {int:Action.Action}): dictionary of actions  
    
    """
    def __init__(self):

        self.actions = {}

    def register(self, action_name, action):
        """Method that registers a new action to the action factory

        Args:
            action_name (str): the name of the ction
            action (Action.Action): the action to be added
        """        
        self.actions[action_name] = action

    def initialize(self, action_name, **kwargs):
        """Method that initializes a new action from the factory

        Args:
            action_name (str): action name
            **kwargs : keywords required by the action constructor

        Raises:
            ValueError: raises a valueError if the action is None

        Returns:
            Action.Action: the action to be returned
        """
        action = self.actions.get(action_name)
        if action is None:
            raise ValueError(action_name)
        return action(**kwargs)


# Factory initialization
Actionf = ActionFactory()
Actionf.register("DoNothing", DoNothing)
Actionf.register("ChangeConfiguration", ChangeConfiguration)
