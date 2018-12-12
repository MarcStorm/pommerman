"""
This module contains the base class for all reward classes
"""

class BaseReward():

    def __init__(self, env):
        """
        Constructor takes the environment as argument
        """
        self.env = env


    def get_reward(self, obs, action):
        """
        A rewards function takes a single observation and the action and returns a scalar
        """
        raise NotImplementedError