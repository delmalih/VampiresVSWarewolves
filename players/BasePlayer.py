# -*- coding: UTF-8 -*-

###########
# Imports #
###########

""" Global """
from easydict import EasyDict as edict
import numpy as np
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
        self.player_id = None
        self.game_state = {}
        self.game_data = {}
        self.start_game()
    
    def start_game(self):
        # Send NME
        player2server.send_NME(self.sock, edict(name = self.name))

        # Getting data
        self.update_game_state(["SET", "HUM", "HME", "MAP"])
    
    def update_game(self):
        self.update_game_state(["UPD"])
    
    def play(self):
        next_actions = self.get_next_actions()
        if next_actions is not None:
            params = edict({
                "movs": [{
                    "x_s": next_action[0][0],
                    "y_s": next_action[0][1],
                    "n": next_action[1],
                    "x_t": next_action[2][0],
                    "y_t": next_action[2][1],
                } for next_action in next_actions]
            })
            player2server.send_MOV(self.sock, params)
        self.update_game()
        self.play()

    def get_next_actions(self):
        return None

    def update_game_state(self, headers_to_get):
        info_received = {}
        while headers_to_get:
            header = server2player.get_header(self.sock)
            print(header)
            if header == "END":
                self.__init__(self.name, self.sock)
                break
            else:
                res = server2player.header2action(header)(self.sock)
            if header in headers_to_get:
                info_received.update(res)
                headers_to_get.remove(header)
            else:
                print("Protocol error !")
        
        self.game_data["height"] = info_received.get("height", self.game_data.get("height"))
        self.game_data["width"] = info_received.get("width", self.game_data.get("width"))
        self.game_data["start_position"] = info_received.get("start_position", self.game_data.get("start_position"))
        self.game_data["commands"] = self.game_data.get("commands", {})
        self.game_data["commands"].update(info_received.get("commands", {}))
        if self.player_id is None:
            n_v = self.game_data["commands"][self.game_data["start_position"]][1]
            self.player_id = 1 if n_v != 0 else 2
        self.game_state = self.commands_to_matrix(self.game_data["commands"], self.game_data["height"], self.game_data["width"])

    def commands_to_matrix(self, commands, height, width):
        matrix = np.zeros((height, width, 3))
        for x, y in commands:
            n_h, n_v, n_w = commands[(x, y)]
            matrix[y, x, 0] = n_h
            matrix[y, x, 1] = n_v
            matrix[y, x, 2] = n_w
        return matrix

    def get_possible_actions(self, game_state, player_id):
        # Get game limits
        H, W = game_state.shape[:2]

        # Get possible actions
        possible_actions = []
        Ys, Xs = np.where(game_state[:, :, player_id])
        for i in range(len(Xs)):
            x, y = Xs[i], Ys[i]
            number = int(game_state[y, x][player_id])
            if number:
                if y > 0:
                    possible_actions.append([((x, y), number, (x, y-1))])
                if y < H - 1:
                    possible_actions.append([((x, y), number, (x, y+1))])

                if x > 0:
                    possible_actions.append([((x, y), number, (x-1, y))])
                    if y > 0:
                        possible_actions.append([((x, y), number, (x-1, y-1))])
                    if y < H - 1:
                        possible_actions.append([((x, y), number, (x-1, y+1))])
                
                if x < W - 1:
                    possible_actions.append([((x, y), number, (x+1, y))])
                    if y > 0:
                        possible_actions.append([((x, y), number, (x+1, y-1))])
                    if y < H - 1:
                        possible_actions.append([((x, y), number, (x+1, y+1))])

        return possible_actions
        