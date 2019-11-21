# -*- coding: UTF-8 -*-

###########
# Imports #
###########

""" Global """
import sys
from easydict import EasyDict as edict

"""" Local """
from BasePlayer import BasePlayer
sys.path.append("..")
from utils import player2server

################
# Player Class #
################

class IbrahimPlayer(BasePlayer):
    def __init__(self, name, sock):
        BasePlayer.__init__(self, name, sock)

    def get_next_move(self):
        # TODO
        return None