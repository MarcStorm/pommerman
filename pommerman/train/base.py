"""
Module implements base utilities for use in the train module
"""

from pommerman import agents, characters
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme
import torch
import os
import datetime
import numpy as np

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


class BaseTraining(object):
    """
    Base training class which sets up the basics for the training environment for a net.
    It sets up the environment

    Note:
        This method should never be instantiated on its own. It is used to inherit from.
    """

    def __init__(self):
        super().__init__()
        self.env = self.set_up_env()
        
    def set_up_env(self):
            # Instantiate the environment
        config = ffa_v0_fast_env()
        env = Pomme(**config["env_kwargs"])

        # Create a set of agents (exactly four)
        agent_list = [
            TrainingAgent(config["agent"](0, config["game_type"])),
            agents.SimpleAgent(config["agent"](1, config["game_type"])),
            agents.SimpleAgent(config["agent"](2, config["game_type"])),
            agents.RandomAgent(config["agent"](3, config["game_type"])),
            # agents.DockerAgent("pommerman/simple-agent", port=12345),
        ]

        env.set_agents(agent_list)
        env.set_training_agent(0) #<- Does not call act method on training agents in env.act
        #env.model = ReinforceModel()
        env.set_init_game_state(None)
        
        return env

    def saveNetwork(self):
        dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        #t = datetime.date.today().strftime("%Y-%m-%d")
        t = datetime.date.today().strftime("%m_%d_%Y")
        filename = "resources/q_agent_{}.pt".format(t)
        PATH = os.path.join(dirpath, filename)
        print (PATH)
        
        torch.save(self.neuralNet.state_dict(), PATH)