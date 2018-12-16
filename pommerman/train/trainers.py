"""
Module implements base utilities for use in the train module
"""
import torch
import os
import datetime
import numpy as np

import datetime
import torch
import json
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
import random

class BaseTraining(object):
    """
    Base training class which sets up the basics for the training environment for a net.
    It sets up the environment

    Note:
        This method should never be instantiated on its own. It is used to inherit from.
    """

    def __init__(self, env, net):
        super().__init__()
        self.env = env
        self.neuralNet = net

    def saveNetwork(self):
        dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        #t = datetime.date.today().strftime("%Y-%m-%d")
        t = datetime.date.today().strftime("%m_%d_%Y")
        filename = "resources/q_agent_{}.pt".format(t)
        PATH = os.path.join(dirpath, filename)
        print (PATH)

        torch.save(self.neuralNet.state_dict(), PATH)


class PolicyTraining(BaseTraining):

    def __init__(self, env, neuralNet, num_episodes, discount_factor, val_freq, visualize=False, reward=None):
        super().__init__(env, neuralNet)
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.val_freq = val_freq
        self.neuralNet = neuralNet
        self.visualize = visualize
        self.reward = reward

    def train(self):
        # train policy network

        try:
            training_rewards, validation_rewards, losses, a_probs_list = [], [], [], []
            epsilon = 1.0
            validation_games = 50
            print('start training')
            for i in range(self.num_episodes):
                rollout = []
                s = self.env.reset()
                done = False
                while(not done):
                    # generate rollout by iteratively evaluating the current policy on the environment
                    with torch.no_grad():
                        a_prob = self.neuralNet(np.atleast_1d(s[0]))
                    a = (np.cumsum(a_prob.numpy()) > np.random.rand()).argmax() # sample action
                    a_probs_list.append(a_prob.numpy()[0].tolist())

                    # sample random action based on epsilon
                    if np.random.rand() < epsilon:
                        a = np.int64(self.env.action_space.sample())

                    actions = self.env.act(s)
                    actions.insert(0,a)

                    obs, reward, done, _ = self.env.step(actions)

                    if self.reward is not None:
                        r = self.reward.get_reward(s[0], a)
                    else:
                        r = reward[0]

                    rollout.append((s[0], a, r))

                    s = obs
                    if done: break

                epsilon *= self.num_episodes / (i / (self.num_episodes / 8) + self.num_episodes)  # decrease epsilon
                # prepare batch
                rollout = np.array(rollout)
                states = np.vstack(rollout[:,0])
                actions = np.vstack(rollout[:,1])
                rewards = np.array(rollout[:,2], dtype=float)
                returns = self.compute_returns(rewards, self.discount_factor)

                # policy gradient update
                self.neuralNet.optimizer.zero_grad()

                # pylint: disable=maybe-no-member
                a_probs = self.neuralNet([s[0] for s in states]).gather(1, torch.from_numpy(actions)).view(-1)
                # pylint: disable=maybe-no-member
                loss = self.neuralNet.loss(a_probs, torch.from_numpy(returns).float())
                loss.backward()
                self.neuralNet.optimizer.step()

                # bookkeeping
                training_rewards.append(sum(rewards))
                losses.append(loss.item())

                # print
                if (i+1) % self.val_freq == 0:
                    # validation
                    print('saving model for iteration: {}'.format(str(i+1)))
                    print('Value of epsilon when saving the model is: {}'.format(str(epsilon)))
                    t = datetime.date.today().strftime("%Y-%m-%d")
                    PATH = "resources/reinforce_agent_{}_{}.pt".format(t,str(i+1))
                    torch.save(self.neuralNet.state_dict(), PATH)
                    
                    for ite in range(validation_games):
                        s = self.env.reset()
                        reward = 0
                        done = False
                        #print("Iteration:",ite)
                        while not done:
                            if self.visualize:
                                self.env.render()
                            with torch.no_grad():
                                a = self.neuralNet(np.atleast_1d(s[0])).float().argmax().item()
                            actions = self.env.act(s)
                            actions.insert(0,a)
                            s, r_all, done, _ = self.env.step(actions)
                            if self.reward is not None:
                                r = self.reward.get_reward(s[0], a)
                            else:
                                r = r_all[0]
                            reward += r
                            
                            if done: break
                        validation_rewards.append(reward)
                    t = datetime.datetime.now()
                    if self.visualize:
                        self.env.render(close=True)
                    print('{:4d}. mean training reward: {:6.2f}, mean validation reward: {:6.2f}, mean loss: {:7.4f}, time: {}'.format(i+1, np.mean(training_rewards[-self.val_freq:]), np.mean(validation_rewards[-validation_games:]), np.mean(losses[-self.val_freq:]), t))
            print('done')
            self.saveList(training_rewards, 'training_rewards_softmax')
            self.saveList(validation_rewards, 'validation_rewards_softmax')
            self.saveList(losses, 'losses_softmax')
            self.saveList(a_probs_list, 'a_probs_list_softmax')
        except KeyboardInterrupt:
            print('interrupt')

    def compute_returns(self, rewards, discount_factor):
        """Compute discounted returns."""
        returns = np.zeros(len(rewards))
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards)-1)):
            returns[t] = rewards[t] + discount_factor * returns[t+1]
        return returns

    def saveList(self, list_to_save, filename):
        with open('resources/' + filename + '.json', 'w+') as f:
            json.dump(list_to_save, f)

class QTraining(BaseTraining):

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
            

    def __init__(self, env, neuralNet_policy, neuralNet_target, num_episodes, val_freq, discount_factor, tau, batch_size, replay_memory_capacity=1000, prefill_memory=True, visualize=False, reward=None):
    
        super().__init__(env, neuralNet_policy)
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.tau = tau
        self.val_freq = val_freq
        self.neuralNet_policy = neuralNet_policy
        self.neuralNet_target = neuralNet_target
        self.visualize = visualize
        self.reward = reward
        self.batch_size = batch_size
        self.replay_memory_capacity = replay_memory_capacity
        self.prefill_memory = prefill_memory 
        self.replay_memory = self.ReplayMemory(self.replay_memory_capacity)
        
        neuralNet_target.load_state_dict(neuralNet_policy.state_dict())
        
        # prefill replay memory with random actions
        if self.prefill_memory:
            print('prefill replay memory')
        
            s = self.env.reset()
            while self.replay_memory.count() < self.replay_memory_capacity:
                a = self.env.act(s)
                a.append(0)
                s1, r, d, _ = self.env.step(a)
                self.replay_memory.add(s[3], a[3], r[3], s1[3], d)
                s = s1 if not d else self.env.reset()
    
    
    def train(self):
        # training loop
        try:
            print('start training')
            epsilon = 1.0
            rewards, lengths, losses, epsilons = [], [], [], []
            for i in range(self.num_episodes):
                s = self.env.reset()
        
                # init new episode
                ep_reward, ep_loss = 0, 0
                d = False
                j = -1
                while not d:
                    j += 1
                    # select action with epsilon-greedy strategy
                    if np.random.rand() < epsilon:
                        a = self.env.action_space.sample()
                    else:
                        with torch.no_grad():
                            a = get_numpy(self.neuralNet_policy(np.atleast_1d(s[3]))).argmax().item()
                    # perform action
                    actions = self.env.act(s)
                    actions.append(a)
                    s1, r, d, _ = self.env.step(actions)
                    # store experience in replay memory
                    self.replay_memory.add(s[3], a, r[3], s1[3], d)
                    # batch update
                    if self.replay_memory.count() >= self.batch_size:
                        # sample batch from replay memory
                        batch = np.array(self.replay_memory.sample(self.batch_size))
                        ss, aa, rr, ss1, dd = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:,4]
                        # do forward pass of batch
                        self.neuralNet_policy.optimizer.zero_grad()
        
                        Q = self.neuralNet_policy(ss)
                        # use target network to compute target Q-values
                        with torch.no_grad():
                            Q1 = self.neuralNet_target(ss1)
                        # compute target for each sampled experience
                        q_targets = Q.clone()
                        for k in range(self.batch_size):
                            q_targets[k, aa[k]] = rr[k] + self.discount_factor * Q1[k].max().item() * (not dd[k])
                        # update network weights
                        loss = self.neuralNet_policy.loss(Q, q_targets)
                        loss.backward()
                        self.neuralNet_policy.optimizer.step()
                        # update target network parameters from policy network parameters
                        self.neuralNet_target.update_params(self.neuralNet_policy.state_dict(), self.tau)
                    else:
                        loss = 0
                    # bookkeeping
                    s = s1
                    ep_reward += r[3]
                    ep_loss += loss.item()
                # bookkeeping
                #epsilon = epsilon
                epsilon *= self.num_episodes/(i/(self.num_episodes/20)+self.num_episodes) # decrease epsilon
                epsilons.append(epsilon); rewards.append(ep_reward); lengths.append(j+1); losses.append(ep_loss)
                if (i+1) % self.val_freq == 0: print('%5d mean training reward: %5.2f' % (i+1, np.mean(rewards[-self.val_freq:])))
            print('done')
        except KeyboardInterrupt:
            print('interrupt')
    

    def compute_returns(self, rewards):
        """Compute discounted returns."""
        returns = np.zeros(len(rewards))
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards)-1)):
            returns[t] = rewards[t] + self.discount_factor * returns[t+1]
        return returns
