

class Constraint:
    """class that defines the differents application constraints
        Args:
            constraints (dict:{str:Constraint.Constraint}): _description_
    """
    def __init__(self, constraints):
        
        self.l_constraints = {}
        for const_name in constraints.keys():
            self.l_constraints[const_name] = constraints[const_name]

    def __str__(self):
        """convert the constraints into a string
        

        Returns:
            str: the string description
        """        
       
        s = '\n"Constraints": {'
        for const_name in self.l_constraints.keys():
            s += '"{}":{},'.format(const_name, self.l_constraints[const_name])
        s = s[:-1] + '}'
        return s
