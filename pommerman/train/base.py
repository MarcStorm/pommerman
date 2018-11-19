"""
Module implements base utilities for use in the train module
"""

from pommerman import agents, characters

class TrainingAgent(agents.BaseAgent):
    """
    A taining agent which actions are given from a neural network

    Note:
        the act() method should not be called since this agent is only
        for training purposes
    """

    def __init__(self, Character=characters.Bomber, *args, **kwargs):
        super().__init__(Character,*args, **kwargs)
        
    
    def act(self, obs, action_space):
        raise Exception('The act function of the training agent should never be called!')