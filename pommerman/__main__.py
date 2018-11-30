#!/usr/bin/bash

from train import PolicyTraining
from train.reward import BlockReward
from neuralnet.convolutional import ConvNet
from environment import stopEnv, randomEnv, simpleEnv

if __name__ == '__main__':
    # Chose a neural network to train
    net = ConvNet()

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Parameters: {}".format(pytorch_total_params))

    # Set up an environment
    env = simpleEnv(static=True)

    # Chose a reward function to tain with (or None for default reward function)
    r = None

    # Initialize a trainer
    trainer = PolicyTraining(env, net, num_episodes=150000, val_freq=25000, discount_factor=0.97, visualize=False, reward=r)

    # Start training the network
    trainer.train()
