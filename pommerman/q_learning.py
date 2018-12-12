import os
import sys
import numpy as np
import torch
import random
import datetime
import util
from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility, characters
from pommerman.constants import Action
from collections import deque
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

use_cuda = False
print('CUDA IS ON: {}'.format(use_cuda))

def get_cuda(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x

def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()

torch.cuda.is_available()

class ReplayMemory(object):
    """Experience Replay Memory"""

    def __init__(self, capacity):
        #self.size = size
        self.memory = deque(maxlen=capacity)

    def add(self, *args):
        """Add experience to memory."""
        self.memory.append([*args])

    def sample(self, batch_size):
        """Sample batch of experiences from memory with replacement."""
        return random.sample(self.memory, batch_size)

    def count(self):
        return len(self.memory)

batch_norm=False
in_channels = 3
out_channels = 3
kernel_size = 5

class DQN(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate):
        super(DQN, self).__init__()
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

        x = util.flatten_state_not_first_board(x)
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

def compute_returns(rewards, discount_factor):
    """Compute discounted returns."""
    returns = np.zeros(len(rewards))
    returns[-1] = rewards[-1]
    for t in reversed(range(len(rewards)-1)):
        returns[t] = rewards[t] + discount_factor * returns[t+1]
    return returns

# one-hot encoder for the states
def one_hot(i, l):
    a = np.zeros((len(i), l))
    a[range(len(i)), i] = 1
    return a

# Instantiate the environment
config = ffa_v0_fast_env()
env = Pomme(**config["env_kwargs"])


#n_inputs = env.observation_space.shape[0]
n_inputs = 372
n_hidden = 500
n_outputs = env.action_space.n

print('state shape:', n_inputs)
print('action shape:', n_outputs)

class TrainingAgent(BaseAgent):

    def __init__(self, character=characters.Bomber):
        super().__init__(character)


    def act(self, obs, action_space):
        return 0

# train Deep Q-network

num_episodes = 5000
batch_size = 64
learning_rate = 0.005
gamma = 0.99 # discount rate
tau = 0.01 # target network update rate
replay_memory_capacity = 10000
prefill_memory = True
val_freq = 10000 # validation frequency

# initialize DQN and replay memory
policy_dqn = DQN(n_inputs, n_hidden, n_outputs, learning_rate)
target_dqn = DQN(n_inputs, n_hidden, n_outputs, learning_rate)
target_dqn.load_state_dict(policy_dqn.state_dict())

if use_cuda:
    policy_dqn.cuda()
    target_dqn.cuda()

replay_memory = ReplayMemory(replay_memory_capacity)

# Add four random agents
agents = []
#for agent_id in range(4):
#    agents[agent_id] = RandomAgent(config["agent"](agent_id, config["game_type"]))
agents = {
    '0' : SimpleAgent(config["agent"](0, config["game_type"])),
    '1' : RandomAgent(config["agent"](1, config["game_type"])),
    '2' : RandomAgent(config["agent"](2, config["game_type"])),
    '3' : TrainingAgent(config["agent"](3, config["game_type"]))
}
env.set_agents(list(agents.values()))
env.set_training_agent(3)
env.set_init_game_state(None)

# prefill replay memory with random actions
if prefill_memory:
    print('prefill replay memory')

    s = env.reset()
    while replay_memory.count() < replay_memory_capacity:
        a = env.act(s)
        a.append(0)
        s1, r, d, _ = env.step(a)
        replay_memory.add(s[3], a[3], r[3], s1[3], d)
        s = s1 if not d else env.reset()

# training loop
try:
    print('start training')
    epsilon = 1.0
    rewards, lengths, losses, epsilons = [], [], [], []
    for i in range(num_episodes):
        s = env.reset()

        # init new episode
        ep_reward, ep_loss = 0, 0
        d = False
        j = -1
        while not d:
            j += 1
            # select action with epsilon-greedy strategy
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = get_numpy(policy_dqn(np.atleast_1d(s[3]))).argmax().item()
            # perform action
            actions = env.act(s)
            actions.append(a)
            s1, r, d, _ = env.step(actions)
            # store experience in replay memory
            replay_memory.add(s[3], a, r[3], s1[3], d)
            # batch update
            if replay_memory.count() >= batch_size:
                # sample batch from replay memory
                batch = np.array(replay_memory.sample(batch_size))
                ss, aa, rr, ss1, dd = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:,4]
                # do forward pass of batch
                policy_dqn.optimizer.zero_grad()

                Q = policy_dqn(ss)
                # use target network to compute target Q-values
                with torch.no_grad():
                    # TODO: use target net
                    Q1 = target_dqn(ss1)
                # compute target for each sampled experience
                q_targets = Q.clone()
                for k in range(batch_size):
                    q_targets[k, aa[k]] = rr[k] + gamma * Q1[k].max().item() * (not dd[k])
                # update network weights
                loss = policy_dqn.loss(Q, q_targets)
                loss.backward()
                policy_dqn.optimizer.step()
                # update target network parameters from policy network parameters
                target_dqn.update_params(policy_dqn.state_dict(), tau)
            else:
                loss = 0
            # bookkeeping
            s = s1
            ep_reward += r[3]
            ep_loss += loss.item()
        # bookkeeping
        #epsilon = epsilon
        epsilon *= num_episodes/(i/(num_episodes/20)+num_episodes) # decrease epsilon
        epsilons.append(epsilon); rewards.append(ep_reward); lengths.append(j+1); losses.append(ep_loss)
        if (i+1) % val_freq == 0: print('%5d mean training reward: %5.2f' % (i+1, np.mean(rewards[-val_freq:])))
    print('done')
except KeyboardInterrupt:
    print('interrupt')

# plot results
def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n
'''
plt.figure(figsize=(16, 9))
plt.subplot(411)
plt.title('training rewards')
plt.plot(range(1, num_episodes+1), rewards)
plt.plot(moving_average(rewards))
plt.xlim([0, num_episodes])
plt.subplot(412)
plt.title('training lengths')
plt.plot(range(1, num_episodes+1), lengths)
plt.plot(range(1, num_episodes+1), moving_average(lengths))
plt.xlim([0, num_episodes])
plt.subplot(413)
plt.title('training loss')
plt.plot(range(1, num_episodes+1), losses)
plt.plot(range(1, num_episodes+1), moving_average(losses))
plt.xlim([0, num_episodes])
plt.subplot(414)
plt.title('epsilon')
plt.plot(range(1, num_episodes+1), epsilons)
plt.xlim([0, num_episodes])
plt.tight_layout(); plt.show()
'''
## Save file
t = datetime.date.today().strftime("%Y-%m-%d")
PATH = "resources/q_agent_{}.pt".format(t)
torch.save(policy_dqn.state_dict(), PATH)
