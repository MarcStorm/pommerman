#!/usr/bin/bash

from train import PolicyTraining, QTraining
from train.reward import BlockReward
from neuralnet.convolutional import ConvNet
from environment import stopEnv, randomEnv, simpleEnv

if __name__ == '__main__':
    # Choose a neural network to train
    net = ConvNet()
    
    # Choose policy and target nets for q learning
    #policy_net = ConvNet()  
    #target_net = ConvNet()
    
    print("Trainable params: {}".format(net.num_trainable_params()))

    # Set up an environment
    env = simpleEnv(static=True)

    # Chose a reward function to tain with (or None for default reward function)
    r = BlockReward(env)

    # Initialize a trainer
    trainer = PolicyTraining(env, net, num_episodes=150000, val_freq=25000, discount_factor=0.9, visualize=False, reward=r)
    #trainer = QTraining(env, policy_net, target_net, num_episodes=2, val_freq=2, discount_factor=0.97, tau=1E-3, batch_size=64, visualize=False, reward=r)
    
    # Start training the network
    trainer.train()
