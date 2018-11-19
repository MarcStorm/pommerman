"""
Module implements a training loop for Q-Learning
"""

import time
import datetime
import pommerman
from util import flatten_state, flatten_state_no_board, flatten_state_not_first_board
from pommerman import agents
from pommerman import constants as c
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility
from pommerman import forward_model
from pommerman import constants

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base import TrainingAgent

class QTraining(object):
    
    def __init__(self, neuralNet, num_episodes, discount_factor, val_freq):
        super().__init__()
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.val_freq = val_freq
        self.env = self.set_up_env()
        #number of actions was set here
        #netKwargs['n_outputs'] = self.env.action_space.n
        # setup policy network
        self.neuralNet = neuralNet
        
    def set_up_env(self):
            # Instantiate the environment
        config = ffa_v0_fast_env()
        env = Pomme(**config["env_kwargs"])

        # Create a set of agents (exactly four)
        agent_list = [
            TrainingAgent(config["agent"](0, config["game_type"])),
            agents.SimpleAgent(config["agent"](1, config["game_type"])),
            agents.SimpleAgent(config["agent"](2, config["game_type"])),
            agents.RandomAgent(config["agent"](3, config["game_type"])),
            # agents.DockerAgent("pommerman/simple-agent", port=12345),
        ]

        env.set_agents(agent_list)
        env.set_training_agent(0) #<- Does not call act method on training agents in env.act
        #env.model = ReinforceModel()
        env.set_init_game_state(None)
        
        return env


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

                    t = datetime.date.today().strftime("%Y-%m-%d")
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
