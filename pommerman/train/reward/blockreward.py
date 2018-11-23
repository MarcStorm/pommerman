from .base import BaseReward
from util import get_valid_actions

import copy
import numpy as np
from pommerman.constants import Item, Action

class BlockReward(BaseReward):
    """
    Block reward gives the agent a reward for destroying blocks within the game
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.agent_id = 0
        self.agents = self.env._agents
        self.agent = self.agents[self.env.training_agent]
        self.last_obs = None


    # Reset the reward object
    def reset(self):
        self.last_obs = None

    def get_board_freq(self, board):
        unique, counts = np.unique(board, return_counts=True)
        items = [Item(i) for i in unique]
        return dict(zip(items, counts))


    def get_reward(self, obs, action):

        reward = 0

        if self.last_obs is None:
            self.last_obs = copy.deepcopy(obs)

        last_freq = self.get_board_freq(self.last_obs['board'])
        curr_freq = self.get_board_freq(obs['board'])
        wood_diff = last_freq[Item.Wood] - curr_freq[Item.Wood]

        reward += 30*wood_diff

        self.last_obs = copy.deepcopy(obs)

        alive_agents = [num for num, agent in enumerate(self.agents) \
                        if agent.is_alive]

        if len(alive_agents) == 0:
            print("TIE!")
            # Game is tie, everyone gets -1.
            return reward - 300
        elif len(alive_agents) == 1:
            # An agent won. Give them +1, others -1.
            if alive_agents[0] == self.agent_id:
                return reward + 100
            else:
                return reward - 300
        elif self.env._step_count > self.env._max_steps:
            # Game is over from time. Everyone gets -1.
            return reward - 300
        else:
            # Game running: 0 for alive, -1 for dead.
            if self.agent.is_alive:
                valid_actions = get_valid_actions(obs)
                if action == Action.Stop:
                    return reward - 1
                if action in valid_actions:
                    return reward + 1
                else:
                    return reward - 2
                return reward
            else:
                return reward - 300