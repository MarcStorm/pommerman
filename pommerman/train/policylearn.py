"""
Module implements a training loop for Policy-based deep reinforcement learning
"""

import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base import BaseTraining

class PolicyTraining(BaseTraining):
    
    def __init__(self, neuralNet, num_episodes, discount_factor, val_freq):
        super().__init__()
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.val_freq = val_freq
        self.neuralNet = neuralNet

    def train(self):
        # train policy network

        try:
            training_rewards, losses = [], []
            print('start training')
            for i in range(self.num_episodes):
                rollout = []
                s = self.env.reset()
                done = False
                while(not done):
                    # generate rollout by iteratively evaluating the current policy on the environment
                    with torch.no_grad():
                        a_prob = self.neuralNet(np.atleast_1d(s[3]))
                    a = (np.cumsum(a_prob.numpy()) > np.random.rand()).argmax() # sample action

                    actions = self.env.act(s)
                    actions.insert(0,a)
                    
                    obs, reward, done, info = self.env.step(actions)
                    
                    rollout.append((s[3], a, reward[3]))
                    
                    s = obs
                    if done: break
                        
                # prepare batch
                rollout = np.array(rollout)
                states = np.vstack(rollout[:,0])
                actions = np.vstack(rollout[:,1])
                rewards = np.array(rollout[:,2], dtype=float)
                returns = self.compute_returns(rewards, self.discount_factor)
                
                # policy gradient update
                self.neuralNet.optimizer.zero_grad()

                a_probs = self.neuralNet([s[0] for s in states]).gather(1, torch.from_numpy(actions)).view(-1)
                loss = self.neuralNet.loss(a_probs, torch.from_numpy(returns).float())
                loss.backward()
                self.neuralNet.optimizer.step()
                
                # bookkeeping
                training_rewards.append(sum(rewards))
                losses.append(loss.item())
                
                # print
                if (i+1) % self.val_freq == 0:
                    # validation
                    validation_rewards = []
                    for _ in range(10):
                        s = self.env.reset()
                        reward = 0
                        done = False
                        while not done:
                            with torch.no_grad():
                                a = self.neuralNet(np.atleast_1d(s[0])).float().argmax().item()
                                
                            actions = self.env.act(obs)
                            actions.insert(0,a)
                            s, r, done, info = self.env.step(actions)
                            reward += r[0]
                            if done: break
                        validation_rewards.append(reward)
                    t = datetime.datetime.now()
                    print('{:4d}. mean training reward: {:6.2f}, mean validation reward: {:6.2f}, mean loss: {:7.4f}, time: {}'.format(i+1, np.mean(training_rewards[-self.val_freq:]), np.mean(validation_rewards), np.mean(losses[-self.val_freq:]), t))
            print('done')
        except KeyboardInterrupt:
            print('interrupt')

    def compute_returns(self, rewards, discount_factor):
        """Compute discounted returns."""
        returns = np.zeros(len(rewards))
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards)-1)):
            returns[t] = rewards[t] + discount_factor * returns[t+1]
        return returns


