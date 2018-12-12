"""
Contains different agents and methods for environments
""" 
from pommerman import agents, characters
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme

from .static import StaticPomme

import torch
import os 

import numpy as np

agent_hooks = {
    'si': 'agents.SimpleAgent(config["agent"](X, config["game_type"]))',
    'st': 'StopAgent(config["agent"](X, config["game_type"]))',
    'ra': 'agents.RandomAgent(config["agent"](X, config["game_type"]))',
}
class TestAgent(agents.BaseAgent):
    """
    A test agent which is used to visualize the result of training a net

    Note:
        takes the net and path to saved weighted
    """

    def __init__(self, neuralnet, path, character=characters.Bomber):
        super().__init__(character)
        self.policyNet = neuralnet
        self.state_list = torch.load(loadNetwork(path))
        self.policyNet.load_state_dict(self.state_list)
        
        
    def act(self, obs, action_space):
        # Kald neuralt netværk og return
        with torch.no_grad():
            a_prob = self.policyNet(np.atleast_1d(obs))
        a = (np.cumsum(a_prob.numpy()) > np.random.rand()).argmax() # sample action
        
        return a

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

class StopAgent(agents.BaseAgent):
    """
    A stop agent which returns the stop action all the time
    """

    def __init__(self, Character=characters.Bomber, *args, **kwargs):
        super().__init__(Character,*args, **kwargs)
        
    
    def act(self, obs, action_space):
        return 0

def loadNetwork(net_name):
        dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filename = "resources/{}.pt".format(net_name)
        PATH = os.path.join(dirpath, filename)
        return PATH

def createEnvironment(func):
    def environment_wrapper(*args, static=False, **kwargs):
        config = ffa_v0_fast_env()
        if static:
            env = StaticPomme(**config["env_kwargs"])
        else:
            env = Pomme(**config["env_kwargs"])
            
            
        if 'net_dict' in kwargs and 'manual_agents' in kwargs:
            agent_list  = func(config,kwargs['net_dict'], kwargs['manual_agents'])

        elif 'manual_agents' in kwargs:
            agent_list  = func(config,kwargs['manual_agents'])

        else:
            agent_list  = func(config)

        if 'test' in kwargs and kwargs['test'] == True:
            env.set_agents(list(list(agent_list.values())))
            env.set_init_game_state(None)
        else:
            env.set_agents(list(agent_list.values()))
            env.set_training_agent(0) #<- Does not call act method on training agents in
            env.set_init_game_state(None)

        return env

    return environment_wrapper

@createEnvironment
def randomEnv(config):
    agent_list = {
        '0' : TrainingAgent(config["agent"](0, config["game_type"])),
        '1' : agents.RandomAgent(config["agent"](1, config["game_type"])),
        '2' : agents.RandomAgent(config["agent"](2, config["game_type"])),
        '3' : agents.RandomAgent(config["agent"](3, config["game_type"])),
    }
    return agent_list

@createEnvironment
def stopEnv(config):
    agent_list = {
        '0' : TrainingAgent(config["agent"](0, config["game_type"])),
        '1' : StopAgent(config["agent"](1, config["game_type"])),
        '2' : StopAgent(config["agent"](2, config["game_type"])),
        '3' : StopAgent(config["agent"](3, config["game_type"])),
    }
    return agent_list

@createEnvironment
def simpleEnv(config):
    agent_list = {
        '0': TrainingAgent(config["agent"](0, config["game_type"])),
        '1': agents.SimpleAgent(config["agent"](1, config["game_type"])),
        '2': agents.SimpleAgent(config["agent"](2, config["game_type"])),
        '3': agents.SimpleAgent(config["agent"](3, config["game_type"])),
    }
    return agent_list

@createEnvironment
def manualEnv(config, manual_agents = ["si", "ra", "ra"]):
    agent_list = [
        TrainingAgent(config["agent"](0, config["game_type"]))]
    for idx, agent in enumerate(manual_agents):
        idx = str(idx+1)
        magic_string = agent_hooks[agent].replace('X', idx)
        agent_list[idx] = eval(magic_string)
    return agent_list

@createEnvironment
def visualEnv(config, net_dict, manual_agents = ["si", "ra", "ra"]):
    agent_list = {
        '0': TestAgent(net_dict['net'], net_dict['filename'], config["agent"](0, config["game_type"]))
    }
    for idx, agent in enumerate(manual_agents):
        idx = str(idx+1)
        magic_string = agent_hooks[agent].replace('X', idx)
        agent_list[idx] = eval(magic_string)
    return agent_list