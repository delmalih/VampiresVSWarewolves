# -*- coding: UTF-8 -*-

###########
# Imports #
###########

""" Global """
from easydict import EasyDict as edict

"""" Local """
from utils import player2server
from utils import server2player

################
# Player Class #
################


class BasePlayer:
    def __init__(self, name, sock):
        self.name = name
        self.sock = sock
        self.game_state = {}
        self.start_game()
    
    def start_game(self):
        # Send NME
        player2server.send_NME(self.sock, edict(name = self.name))

        # Getting data
        self.update_game_state(["SET", "HUM", "HME", "MAP"])
    
    def update_game(self):
        self.update_game_state(["UPD"])
    
    def play(self):
        pass

    def update_game_state(self, headers_to_get):
        while headers_to_get:
            header = server2player.get_header(self.sock)
            if header in headers_to_get:
                res = server2player.header2action(header)(self.sock)
                self.game_state.update(res)
                headers_to_get.remove(header)
            else:
                print("Protocol error !")
        
