'''An example to show how to set up an pommerman game programmatically'''
import copy

# Notebook 6.3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util import flatten_state_not_first_board, get_valid_actions
from pommerman import characters
from pommerman.agents import SimpleAgent, RandomAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.constants import Item
from pommerman.envs.v0 import Pomme

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
                valid_actions = get_valid_actions(obs)
                if action in valid_actions:
                    return reward - 1
                else:
                    return reward - 2
                return reward
            else:
                return reward - 300


class TrainingAgent(BaseAgent):

    def __init__(self, character=characters.Bomber):
        super().__init__(character)


    def act(self, obs, action_space):
        return 0


# Print all possible environments in the Pommerman registry
#print(pommerman.REGISTRY)

# Instantiate the environment
config = ffa_v0_fast_env()
env = Pomme(**config["env_kwargs"])

batch_norm=False
in_channels = 3
out_channels = 3
kernel_size = 5

class PolicyNet(nn.Module):
    """Policy network"""

    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate):
        super(PolicyNet, self).__init__()
        # network
        self.other_shape = [3]

        #Input for conv2d is (batch_size, num_channels, width, height)
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=out_channels,
                               kernel_size=kernel_size, stride=1, padding=2)

        self.conv2 = nn.Conv2d(in_channels = in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=1, padding=2)

        self.conv3 = nn.Conv2d(in_channels = in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=1, padding=2)

        self.convolution_out_size = 11*11*3

        #self.ffn_input_size = n_inputs
        self.ffn_input_size = out_channels * 11 * 11 + 251


        self.ffn = nn.Sequential(
            nn.Linear(self.ffn_input_size, n_hidden),
            nn.ReLU(),
            #
            nn.Dropout(0.25),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(n_hidden, n_outputs),
        )

        self.activation = F.relu

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(11)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x
            self.bn3 = lambda x: x

        self.ffn.apply(self.init_weights)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)


    def forward(self, x):
        board = x[0]['board']

        board = torch.tensor(board)
        board = board.unsqueeze(0)
        board = board.unsqueeze(0)
        board = board.float()
        for i in range(1,len(x)):
            completeBoard = torch.tensor(x[i]['board'])
            completeBoard = completeBoard.unsqueeze(0)
            completeBoard = completeBoard.unsqueeze(0)
            completeBoard = completeBoard.float()
            board = torch.cat([board, completeBoard], dim=0)

        board = torch.autograd.Variable(board)
        board = self.conv1(board)
        board = self.bn1(board)
        board = self.activation(board)
        board = self.conv2(board)
        board = self.bn1(board)
        board = self.activation(board)
        board = self.conv3(board)
        board = self.bn1(board)
        board = self.activation(board)

        x2 = board.view(-1, self.convolution_out_size)

        x = flatten_state_not_first_board(x)
        x = torch.cat([x2, x], dim=1)

        x = self.ffn(x)

        x = F.softmax(x, dim=1)
        return x

    def loss(self, action_probabilities, returns):
        return -torch.mean(torch.mul(torch.log(action_probabilities), returns))

    def init_weights(m, *args):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    # Function from old network
    def update_params(self, new_params, tau):
        params = self.state_dict()
        for k in params.keys():
            params[k] = (1-tau) * params[k] + tau * new_params[k]
        self.load_state_dict(params)

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

num_episodes = 5000
#rollout_limit = env.spec.timestep_limit # max rollout length
discount_factor = 1.0 # reward discount factor (gamma), 1.0 = no discount
learning_rate = 0.001 # you know this by now
val_freq = 25 # validation frequency

# setup policy network

policy = PolicyNet(n_inputs, n_hidden, n_outputs, learning_rate)

if use_cuda:
    policy.cuda()

agents = []
#for agent_id in range(4):
#    agents[agent_id] = RandomAgent(config["agent"](agent_id, config["game_type"]))
agents = {
    '0' : TrainingAgent(config["agent"](0, config["game_type"])),
    '1' : SimpleAgent(config["agent"](1, config["game_type"])),
    '2' : RandomAgent(config["agent"](2, config["game_type"])),
    '3' : RandomAgent(config["agent"](3, config["game_type"]))
}
env.set_agents(list(agents.values()))
env.set_training_agent(0) #<- Does not call act method on training agents in env.act
env.set_init_game_state(None)

# train policy network

try:
    print('start training')
    epsilon = 1.0
    rewards, losses, epsilons = [], [], []
    for i in range(num_episodes):
        rollout = []
        s = env.reset()
        done = False
        ep_reward, ep_loss = 0, 0
        #policy.train()
        while not done:
            # select action with epsilon-greedy strategy
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                # generate rollout by iteratively evaluating the current policy on the environment
                with torch.no_grad():
                    a = policy(np.atleast_1d(s[0])).argmax().item()
            actions = env.act(s)
            actions.append(a)

            s1, r, done, _ = env.step(actions)
            rollout.append((s[0], a, r[0]))
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
        #training_rewards.append(sum(rewards))
        #losses.append(loss.item())
        #policy.eval()

        #epsilon = epsilon
        epsilon *= num_episodes/(i/(num_episodes/20)+num_episodes) # decrease epsilon
        #epsilons.append(epsilon); losses.append(loss)

        # print
        if (i+1) % val_freq == 0:
            # validation
            validation_rewards = []
            for _ in range(3):
                s = env.reset()
                reward = 0
                done = False
                while not done:
                    env.render()
                    with torch.no_grad():
                        a_prob = policy(np.atleast_1d(s[0]))
                    a = (np.cumsum(get_numpy(a_prob)) > np.random.rand()).argmax() # sample action
                    actions = env.act(s)
                    actions.insert(0,a)

                    s, r, done, _ = env.step(actions)
                    #r = policy.get_reward(s[0], Action(a))
                    reward += r[0]
                validation_rewards.append(reward)
                print(reward)
                env.render(close=True)
            print('{:4d}. mean training reward: {:6.2f}, mean validation reward: {:6.2f}, mean loss: {:7.4f}'.format(i+1, np.mean(rewards[-val_freq:]), np.mean(validation_rewards), np.mean(losses[-val_freq:])))
    env.close()
    print('done')
except KeyboardInterrupt:
    print('interrupt')
