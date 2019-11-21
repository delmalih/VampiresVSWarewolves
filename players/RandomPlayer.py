# -*- coding: UTF-8 -*-

###########
# Imports #
###########

""" Global """
import numpy as np

"""" Local """
from .BasePlayer import BasePlayer
from ..utils import player2server

################
# Player Class #
################

class RandomPlayer(BasePlayer):
    def __init__(self, name, sock):
        BasePlayer.__init__(self, name, sock)
    
    def play(self):
        params = {}
        # TODO
        # player2server.send_MOV(self.sock, params)
        self.update_game()
    