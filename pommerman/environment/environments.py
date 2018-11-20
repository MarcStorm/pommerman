"""
Contains different agents and methods for environments
""" 
from pommerman import agents, characters
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme

agent_hooks = {
    'si': 'agents.SimpleAgent(config["agent"](1, config["game_type"]))',
    'st': 'StopAgent(config["agent"](1, config["game_type"]))',
    'ra': 'agents.RandomAgent(config["agent"](1, config["game_type"]))',
}

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

def createEnvironment(func):
    def environment_wrapper(*args, **kwargs):
        config = ffa_v0_fast_env()
        env = Pomme(**config["env_kwargs"])
        if (kwargs):
            agent_list  = func(config,kwargs['manual_agents'])
        else:
            agent_list  = func(config)   
        print (agent_list)
        env.set_agents(agent_list)
        env.set_training_agent(0) #<- Does not call act method on training agents in
        env.set_init_game_state(None)
        return env
    return environment_wrapper

@createEnvironment
def randomEnv(config):
    agent_list = [
        TrainingAgent(config["agent"](0, config["game_type"])),
        agents.RandomAgent(config["agent"](1, config["game_type"])),
        agents.RandomAgent(config["agent"](2, config["game_type"])),
        agents.RandomAgent(config["agent"](3, config["game_type"])),
    ]
    return agent_list

@createEnvironment
def stopEnv(config):
    agent_list = [
        TrainingAgent(config["agent"](0, config["game_type"])),
        agents.StopAgent(config["agent"](1, config["game_type"])),
        agents.StopAgent(config["agent"](2, config["game_type"])),
        agents.StopAgent(config["agent"](3, config["game_type"])),
    ]
    return agent_list

@createEnvironment
def simpleEnv(config):
    agent_list = [
        TrainingAgent(config["agent"](0, config["game_type"])),
        agents.SimpleAgent(config["agent"](1, config["game_type"])),
        agents.SimpleAgent(config["agent"](2, config["game_type"])),
        agents.SimpleAgent(config["agent"](3, config["game_type"])),
    ]
    return agent_list

@createEnvironment
def manualEnv(config, manual_agents = ["si", "st", "ra"]):
    agent_list = [
        TrainingAgent(config["agent"](0, config["game_type"]))]
    for agent in manual_agents:
        agent_list.append(eval(agent_hooks[agent]))
    return agent_list