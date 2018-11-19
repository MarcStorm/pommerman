"""
Module implements a training loop for value-based deep reinforcement learning
"""

import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base import BaseTraining

class QTraining(BaseTraining):
    
    def __init__(self, neuralNet, num_episodes, discount_factor, val_freq):
        super().__init__()
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.val_freq = val_freq
        self.neuralNet = neuralNet

    def train(self):
        try:
            training_rewards, losses = [], []
            print('start training')
            for i in range(self.num_episodes):
                rollout = []
                s = self.env.reset()
                done = False
                #policy.train()
                while not done:
                    # generate rollout by iteratively evaluating the current policy on the environment
                    with torch.no_grad():
                        a_prob = self.neuralNet(np.atleast_1d(s[0]))
                        #a_prob = policy(s[0])
                        #print(s[0])
                    a = (np.cumsum(a_prob.numpy()) > np.random.rand()).argmax() # sample action

                    actions = self.env.act(s)
                    actions.insert(0,a)

                    #print(actions)

                    s1, r, done, _ = self.env.step(actions)
                    #print(r)
                    rollout.append((s[0], a, r[0]))
                    #print("\n\nrollout:",rollout,"\n\n")
                    s = s1
                # prepare batch
                if(i % 10 == 0):
                    print('done with episode:',i)
                rollout = np.array(rollout)
                states = np.vstack(rollout[:,0])
                actions = np.vstack(rollout[:,1])
                rewards = np.array(rollout[:,2], dtype=float)
                returns = self.compute_returns(rewards)
                # policy gradient update
                self.neuralNet.optimizer.zero_grad()
                # pylint: disable=maybe-no-member
                a_probs = self.neuralNet([s[0] for s in states]).gather(1, torch.from_numpy(actions)).view(-1)
                loss = self.neuralNet.loss(a_probs, torch.from_numpy(returns).float())
                loss.backward()
                self.neuralNet.optimizer.step()
                # bookkeeping
                training_rewards.append(sum(rewards))
                losses.append(loss.item())
                #policy.eval()
                # print
                if (i+1) % self.val_freq == 0:
                    # validation
                    validation_rewards = []
                    for _ in range(10):
                        s = self.env.reset()
                        reward = 0
                        done = False
                        while not done:
                            #env.render()
                            with torch.no_grad():
                                probs = self.neuralNet(np.atleast_1d(s[0]))
                                #a_prob = policy(s[0])
                                a = probs.argmax().item()
                                #print(probs, "max actions: ", a,probs.argmax())

                            actions = self.env.act(s)
                            actions.insert(0,a)

                            s, r, done, _ = self.env.step(actions)
                            reward += r[0]
                        validation_rewards.append(reward)
                        #env.render(close=True)

                    t = datetime.datetime.now()
                    print('{:4d}. mean training reward: {:6.2f}, mean validation reward: {:6.2f}, mean loss: {:7.4f}, time:{}'.format(i+1, np.mean(training_rewards[-self.val_freq:]), np.mean(validation_rewards), np.mean(losses[-self.val_freq:]), t))
            self.env.close()
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