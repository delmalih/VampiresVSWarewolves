# -*- coding: UTF-8 -*-

###########
# Imports #
###########

""" Global """
from easydict import EasyDict as edict
import sys

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
        self.player_id = 1
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
        info_received = {}
        while headers_to_get:
            header = server2player.get_header(self.sock)
            res = server2player.header2action(header)(self.sock)
            if header in headers_to_get:
                info_received.update(res)
                headers_to_get.remove(header)
            else:
                print("Protocol error !")
        
        self.game_state["height"] = info_received.get("height", self.game_state.get("height"))
        self.game_state["width"] = info_received.get("width", self.game_state.get("width"))
        self.game_state["houses"] = info_received.get("houses", self.game_state.get("houses"))
        self.game_state["commands"] = self.game_state.get("commands", {})
        self.game_state["commands"].update(info_received.get("commands", {}))

    def get_possible_moves(self):
        H = self.game_state["height"]
        W = self.game_state["width"]

        possible_moves = []
        for x, y in self.game_state["commands"]:
            source = (x, y)
            number = self.game_state["commands"][source][self.player_id]
            if number:
                if y > 0:
                    possible_moves.append((source, number, (x, y-1)))
                if y < H - 1:
                    possible_moves.append((source, number, (x, y+1)))

                if x > 0:
                    possible_moves.append((source, number, (x-1, y)))
                    if y > 0:
                        possible_moves.append((source, number, (x-1, y-1)))
                    if y < H - 1:
                        possible_moves.append((source, number, (x-1, y+1)))
                
                if x < W - 1:
                    possible_moves.append((source, number, (x+1, y)))
                    if y > 0:
                        possible_moves.append((source, number, (x+1, y-1)))
                    if y < H - 1:
                        possible_moves.append((source, number, (x+1, y+1)))

        return possible_moves
        
