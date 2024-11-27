class Transition(object):
    """
    Class to represent a parsing transition in a dependency parser.

    Attributes:
    - action (str): The action to take, represented as an string constant. Actions include SHIFT, REDUCE, LEFT-ARC, or RIGHT-ARC.
    - dependency (str): The type of dependency relationship (only for LEFT-ARC and RIGHT-ARC, otherwise it'll be None), corresponding to the deprel column
    """

    def __init__(self, action: int, dependency: str = None):
        self._action = action
        self._dependency = dependency

    @property
    def action(self):
        """Return the action attribute."""
        return self._action

    @property
    def dependency(self):
        """Return the dependency attribute."""
        return self._dependency

    def __str__(self):
        return f"{self._action}-{self._dependency}" if self._dependency else str(self._action)