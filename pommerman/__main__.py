#!/usr/bin/bash

from train import PolicyTraining
from train.reward import BlockReward
from neuralnet.convolutional import ConvNet
from environment import stopEnv, randomEnv, simpleEnv

if __name__ == '__main__':
    # Chose a neural network to train
    net = ConvNet()

    # Set up an environment
    env = simpleEnv()

    # Chose a reward function to tain with (or None for default reward function)
    r = None

    # Initialize a trainer
    trainer = PolicyTraining(env, net, num_episodes=500, val_freq=100, discount_factor=0.99, visualize=True, reward=r)

    # Start training the network
    trainer.train()
