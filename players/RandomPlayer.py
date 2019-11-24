# -*- coding: UTF-8 -*-

###########
# Imports #
###########

""" Global """
import sys
import numpy as np
from easydict import EasyDict as edict
import time

"""" Local """
from BasePlayer import BasePlayer
sys.path.append("..")
from utils import player2server

################
# Player Class #
################

class RandomPlayer(BasePlayer):
    def __init__(self, name, sock):
        BasePlayer.__init__(self, name, sock)

    def get_next_actions(self):
        possible_actions = self.get_possible_actions(self.game_state, self.player_id)
        if len(possible_actions) > 0:
            next_actions = possible_actions[np.random.choice(range(len(possible_actions)))]
            return next_actions
        return None
    