"""
Contains different agents and methods for environments
""" 
from pommerman import agents, characters
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme

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

def manualEnv(agents):
    """
    A method that returns an env with one training agent and three stop agents defined 
    by a list that is passed to the method
    """
    
    env.set_agents(agent_list)
    env.set_training_agent(0) #<- Does not call act method on training agents in env.act
    #env.model = ReinforceModel()
    env.set_init_game_state(None)
        
    return env  

def randomEnv():
    """
    A method that returns an env with one training agent and three stop agents
    """
    # Instantiate the environment
    config = ffa_v0_fast_env()
    env = Pomme(**config["env_kwargs"])

    # Create a set of agents (exactly four)
    agent_list = [
        TrainingAgent(config["agent"](0, config["game_type"])),
        agents.RandomAgent(config["agent"](1, config["game_type"])),
        agents.RandomAgent(config["agent"](2, config["game_type"])),
        agents.RandomAgent(config["agent"](3, config["game_type"])),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]

    env.set_agents(agent_list)
    env.set_training_agent(0) #<- Does not call act method on training agents in env.act
    #env.model = ReinforceModel()
    env.set_init_game_state(None)
        
    return env

def simpleEnv():
    """
    A method that returns an env with one training agent and three stop agents
    """
        # Instantiate the environment
    config = ffa_v0_fast_env()
    env = Pomme(**config["env_kwargs"])

    # Create a set of agents (exactly four)
    agent_list = [
        TrainingAgent(config["agent"](0, config["game_type"])),
        agents.SimpleAgent(config["agent"](1, config["game_type"])),
        agents.SimpleAgent(config["agent"](2, config["game_type"])),
        agents.SimpleAgent(config["agent"](3, config["game_type"])),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]

    env.set_agents(agent_list)
    env.set_training_agent(0) #<- Does not call act method on training agents in env.act
    #env.model = ReinforceModel()
    env.set_init_game_state(None)
        
    return env

def stopEnv():
    """
    A method that returns an env with one training agent and three stop agents
    """
        # Instantiate the environment
    config = ffa_v0_fast_env()
    env = Pomme(**config["env_kwargs"])

    # Create a set of agents (exactly four)
    agent_list = [
        TrainingAgent(config["agent"](0, config["game_type"])),
        StopAgent(config["agent"](1, config["game_type"])),
        StopAgent(config["agent"](2, config["game_type"])),
        StopAgent(config["agent"](3, config["game_type"])),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]

    env.set_agents(agent_list)
    env.set_training_agent(0) #<- Does not call act method on training agents in env.act
    #env.model = ReinforceModel()
    env.set_init_game_state(None)
        
    return env

def createEnvironment(func):
    def newEnvironment():
        func()

    return newEnvironment