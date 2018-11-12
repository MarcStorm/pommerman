import util

'''An example to show how to set up an pommerman game programmatically'''
import time
import copy
import pommerman
from util import flatten_state
from pommerman import agents
from pommerman import constants as c
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility
from pommerman import forward_model
from pommerman import constants
from pommerman.constants import Action, Item

# Notebook 6.3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
print("Cuda:",use_cuda)

def get_cuda(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x

def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().numpy()
    return x.numpy()


class Reward(object):
    def __init__(self, env, agent_list, agent_id=0):
        self.agents = agent_list
        self.env = env
        self.agent_id = agent_id
        self.agent = agent_list[agent_id]
        self.last_obs = None

    # Reset the reward object
    def reset(self):
        self.last_obs = None

    def get_board_freq(self, board):
        unique, counts = np.unique(board, return_counts=True)
        items = [Item(i) for i in unique]
        return dict(zip(items, counts))

    # Return the reward for an agent given the current observation
    def get_reward(self, obs, action):

        reward = 0

        if self.last_obs is None:
            self.last_obs = copy.deepcopy(obs)

        last_freq = self.get_board_freq(self.last_obs['board'])
        curr_freq = self.get_board_freq(obs['board'])
        wood_diff = last_freq[Item.Wood] - curr_freq[Item.Wood]

        reward += 30*wood_diff

        self.last_obs = copy.deepcopy(obs)

        alive_agents = [num for num, agent in enumerate(self.agents) \
                        if agent.is_alive]

        if len(alive_agents) == 0:
            print("TIE!")
            # Game is tie, everyone gets -1.
            return reward - 300
        elif len(alive_agents) == 1:
            # An agent won. Give them +1, others -1.
            if alive_agents[0] == self.agent_id:
                return reward + 100
            else:
                return reward - 300
        elif self.env._step_count > self.env._max_steps:
            # Game is over from time. Everyone gets -1.
            return reward - 300
        else:
            # Game running: 0 for alive, -1 for dead.
            if self.agent.is_alive:
                valid_actions = util.get_valid_actions(obs)
                if action in valid_actions:
                    return reward - 1
                else:
                    return reward - 2
                return reward
            else:
                return reward - 300


class NewAgent(agents.BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def __init__(self, Character=Bomber, *args, **kwargs):
        super(NewAgent,self).__init__(Character,*args, **kwargs)
        self.seq = [c.Action.Right, c.Action.Up, c.Action.Left, c.Action.Down]
        self.index = 0
    
    def act(self, obs, action_space):
        if self.index == 4:
            self.index = 0
        action = self.seq[self.index]
        self.index += 1
        return 0

class StopAgent(agents.BaseAgent):

    def __init__(self, Character=Bomber, *args, **kwargs):
        super(StopAgent,self).__init__(Character,*args, **kwargs)
    
    def act(self, obs, action_space):
        return 0

    
# Print all possible environments in the Pommerman registry
print(pommerman.REGISTRY)

# Instantiate the environment
config = ffa_v0_fast_env()
env = Pomme(**config["env_kwargs"])

agent = NewAgent(config["agent"](0, config["game_type"]))

# Create a set of agents (exactly four)
agent_list = [
    agent,
    StopAgent(config["agent"](1, config["game_type"])),
    StopAgent(config["agent"](2, config["game_type"])),
    StopAgent(config["agent"](3, config["game_type"])),
]

env.set_agents(agent_list)
env.set_training_agent(0) #<- Does not call act method on training agents in env.act
env.set_init_game_state(None)

class PolicyNet(nn.Module):
    """Policy network"""

    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate):
        super(PolicyNet, self).__init__()
        self.reward = Reward(env, agent_list)

        # Network
        self.ffn = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
            #nn.BatchNorm1d(n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
            #nn.BatchNorm1d(n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            #nn.BatchNorm1d(n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
            #nn.BatchNorm1d(n_hidden),
            nn.Linear(n_hidden, n_outputs),
        )
        
        self.ffn.apply(self.init_weights)
        
        # training
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = util.flatten_state(x)
        x = get_cuda(x)
        x = self.ffn(x)
        return F.softmax(x, dim=1)
    
    def loss(self, action_probabilities, returns):
        return -torch.mean(torch.mul(torch.log(action_probabilities), returns))
    
    def init_weights(m, *args):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def get_reward(self, obs, action):
        return self.reward.get_reward(obs, action)

    
def compute_returns(rewards, discount_factor):
    """Compute discounted returns."""
    returns = np.zeros(len(rewards))
    returns[-1] = rewards[-1]
    for t in reversed(range(len(rewards)-1)):
        returns[t] = rewards[t] + discount_factor * returns[t+1]
    return returns


# training settings
n_inputs = 372
n_hidden = 500
n_outputs = env.action_space.n

num_episodes = 250
#rollout_limit = env.spec.timestep_limit # max rollout length
discount_factor = 1.0 # reward discount factor (gamma), 1.0 = no discount
learning_rate = 0.001 # you know this by now
val_freq = 25 # validation frequency

# setup policy network

policy = PolicyNet(n_inputs, n_hidden, n_outputs, learning_rate)

if use_cuda:
    policy.cuda()

# train policy network

try:
    training_rewards, losses = [], []
    print('start training')
    for i in range(num_episodes):
        rollout = []
        s = env.reset()
        policy.reward.reset()
        done = False
        #policy.train()
        while not done:
            # generate rollout by iteratively evaluating the current policy on the environment
            with torch.no_grad():
                a_prob = policy(np.atleast_1d(s[0]))
            a = (np.cumsum(get_numpy(a_prob)) > np.random.rand()).argmax() # sample action
            actions = env.act(s)
            actions.insert(0,a)
            
            s1, _, done, _ = env.step(actions)
            r = policy.get_reward(s[0], Action(a))
            rollout.append((s[0], a, r))
            s = s1
        # prepare batch
        print('done with episode:',i)
        rollout = np.array(rollout)
        states = np.vstack(rollout[:,0])
        actions = np.vstack(rollout[:,1])
        rewards = np.array(rollout[:,2], dtype=float)
        returns = compute_returns(rewards, discount_factor)
        # policy gradient update
        policy.optimizer.zero_grad()
        a_probs = policy([s[0] for s in states]).cpu().gather(1, torch.from_numpy(actions)).view(-1)
        loss = policy.loss(a_probs, torch.from_numpy(returns).float())
        loss.backward()
        policy.optimizer.step()
        # bookkeeping
        training_rewards.append(sum(rewards))
        losses.append(loss.item())
        #policy.eval()
        # print
        if (i+1) % val_freq == 0:
            # validation
            validation_rewards = []
            for _ in range(3):
                s = env.reset()
                policy.reward.reset()
                reward = 0
                done = False
                while not done:
                    env.render()
                    with torch.no_grad():
                        a_prob = policy(np.atleast_1d(s[0]))
                    a = (np.cumsum(get_numpy(a_prob)) > np.random.rand()).argmax() # sample action
                    actions = env.act(s)
                    actions.insert(0,a)
                    
                    s, _, done, _ = env.step(actions)
                    r = policy.get_reward(s[0], Action(a))
                    reward += r
                validation_rewards.append(reward)
                print(reward)
                env.render(close=True)
            print('{:4d}. mean training reward: {:6.2f}, mean validation reward: {:6.2f}, mean loss: {:7.4f}'.format(i+1, np.mean(training_rewards[-val_freq:]), np.mean(validation_rewards), np.mean(losses[-val_freq:])))
    env.close()
    print('done')
except KeyboardInterrupt:
    print('interrupt')