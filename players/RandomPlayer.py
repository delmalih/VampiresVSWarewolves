# -*- coding: UTF-8 -*-

###########
# Imports #
###########

""" Global """
import sys
import numpy as np
from easydict import EasyDict as edict

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

    def play(self):
        possible_moves = self.get_possible_moves()
        if len(possible_moves) > 0:
            next_moves = [possible_moves[np.random.choice(range(len(possible_moves)))]]
            params = edict({
                "movs": [{
                    "x_s": next_move[0][0],
                    "y_s": next_move[0][1],
                    "n": next_move[1],
                    "x_t": next_move[2][0],
                    "y_t": next_move[2][1],
                } for next_move in next_moves]
            })
            player2server.send_MOV(self.sock, params)
        self.update_game()
        self.play()
    