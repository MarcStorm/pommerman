"""
Module implements base utilities for use in the train module
"""
import torch
import os
import datetime
import numpy as np

import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
            training_rewards, losses = [], []
            epsilon = 1.0
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

                epsilon *= self.num_episodes / (i / (self.num_episodes / 20) + self.num_episodes)  # decrease epsilon
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
                    validation_rewards = []
                    for _ in range(10):
                        s = self.env.reset()
                        reward = 0
                        done = False
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

class QTraining(BaseTraining):

    def __init__(self, env, neuralNet, num_episodes, discount_factor, val_freq):
        super().__init__(env, neuralNet)
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
                self.neuralNet.train()
                while not done:
                    # generate rollout by iteratively evaluating the current policy on the environment
                    with torch.no_grad():
                        a_prob = self.neuralNet(np.atleast_1d(s[0]))
                    a = (np.cumsum(a_prob.numpy()) > np.random.rand()).argmax() # sample action

                    actions = self.env.act(s)
                    actions.insert(0,a)

                    s1, r, done, _ = self.env.step(actions)
                    rollout.append((s[0], a, r[0]))
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
                self.neuralNet.eval()
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
