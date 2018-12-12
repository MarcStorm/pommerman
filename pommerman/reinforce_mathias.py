'''An example to show how to set up an pommerman game programmatically'''
import copy

# Notebook 6.3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
from util import flatten_state_not_first_board, get_valid_actions
from pommerman import characters
from pommerman.agents import SimpleAgent, RandomAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.constants import Item
from pommerman.envs.v0 import Pomme
from pommerman import forward_model
from environment import stopEnv, randomEnv, simpleEnv

use_cuda = torch.cuda.is_available()
print("Cuda:", use_cuda)

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


def compute_returns(rewards, discount_factor):
    """Compute discounted returns."""
    returns = np.zeros(len(rewards))
    returns[-1] = rewards[-1]
    for t in reversed(range(len(rewards)-1)):
        returns[t] = rewards[t] + discount_factor * returns[t+1]
    return returns


# Initialise the training environment
env = simpleEnv(static=True)

num_episodes = 75000
discount_factor = 1 # reward discount factor (gamma), 1.0 = no discount
learning_rate = 0.1 # you know this by now
val_freq = 15 # validation frequency

# setup policy network
from neuralnet.convolutional import ConvNet
policy = ConvNet()

# train policy network
try:
    training_rewards, losses = [], []
    epsilon = 1.0
    print('start training')
    for i in range(num_episodes):
        #print(epsilon)
        rollout = []
        s = env.reset()
        done = False
        while not done:

            with torch.no_grad():
                a_prob = policy(np.atleast_1d(s[0]))
                #print("a_prob: {}".format(a_prob))
            a = (np.cumsum(a_prob.numpy()) > np.random.rand()).argmax() # sample action
            # sample random action based on epsilon
            if np.random.rand() < epsilon:
                a = np.int64(env.action_space.sample())

            #with torch.no_grad():
            #    a_prob = policy(np.atleast_1d(s[0]))

            #a = (np.cumsum(a_prob.numpy()) > np.random.rand()).argmax()  # sample action
            #if np.random.rand() < epsilon:
            #    a = np.int64(env.action_space.sample())
            #    while a_prob[0][a] < 0.01:
            #        a = np.int64(env.action_space.sample())

            actions = env.act(s)
            actions.insert(0, a)

            s1, r, done, _ = env.step(actions)
            rollout.append((s[0], a, r[0]))
            s = s1
        #epsilon *= num_episodes / (i / (num_episodes / 20) + num_episodes)  # decrease epsilon
        epsilon *= num_episodes / (i / (num_episodes / 2) + num_episodes)  # decrease epsilon
        # prepare batch

        rollout = np.array(rollout)
        states = np.vstack(rollout[:, 0])
        actions = np.vstack(rollout[:, 1])
        rewards = np.array(rollout[:, 2], dtype=float)
        returns = compute_returns(rewards, discount_factor)
        # policy gradient update
        policy.optimizer.zero_grad()
        a_probs = policy([s[0] for s in states]).gather(1, torch.from_numpy(actions)).view(-1)
        loss = policy.loss(a_probs, torch.from_numpy(returns).float())
        loss.backward()
        policy.optimizer.step()
        # bookkeeping
        training_rewards.append(sum(rewards))
        losses.append(loss.item())
        if (i + 1) % val_freq == 0:
            print('saving model for iteration: {}'.format(str(i+1)))
            print('Value of epsilon when saving the model is: {}'.format(str(epsilon)))
            t = datetime.date.today().strftime("%Y-%m-%d")
            PATH = "resources/reinforce_agent_{}_{}.pt".format(t,str(i+1))
            torch.save(policy.state_dict(), PATH)
            # validation
            validation_rewards = []
            for _ in range(10):
                s = env.reset()
                reward = 0
                done = False
                while not done:
                    env.render()
                    with torch.no_grad():
                        probs = policy(np.atleast_1d(s[0]))
                        a = probs.argmax().item()
                    actions = env.act(s)
                    actions.insert(0, a)
                    s, r, done, _ = env.step(actions)
                    reward += r[0]
                validation_rewards.append(reward)
            print('{:4d}. mean training reward: {:6.2f}, mean validation reward: {:6.2f}, mean loss: {:7.4f}'.format(
                i + 1, np.mean(training_rewards[-val_freq:]), np.mean(validation_rewards), np.mean(losses[-val_freq:])))
    env.close()
    print('done')
except KeyboardInterrupt:
    print('interrupt')


## Save file
t = datetime.date.today().strftime("%Y-%m-%d")
PATH = "resources/reinforce_agent_{}.pt".format(t)
torch.save(policy.state_dict(), PATH)
