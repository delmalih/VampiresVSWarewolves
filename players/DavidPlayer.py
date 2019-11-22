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

class DavidPlayer(BasePlayer):
    def __init__(self, name, sock):
        BasePlayer.__init__(self, name, sock)

    def get_next_actions(self):
        return self.best_move(self.player_id)
    
    ######################
    # Heuristic function #
    ######################

    def heuristic_value(self, game_state):
        player_id = self.player_id
        ennemy_id = 3 - player_id
        total_population = self.get_total_population(game_state)
        player_population = 1. * self.get_total_population(game_state, filter_id=player_id) / total_population
        ennemy_population = 1. * self.get_total_population(game_state, filter_id=ennemy_id) / total_population
        return player_population * 100 - ennemy_population * 10

    def get_total_population(self, game_state, filter_id=None):
        total = 0
        for coord in game_state["commands"]:
            if filter_id is not None:
                total += game_state["commands"][coord][filter_id]
            else:
                total += sum(game_state["commands"][coord])
        return total

    #####################
    # MiniMax algorithm #
    #####################

    def best_move(self, player_id, depth=3):
        possible_actions = self.get_possible_actions(self.game_state)
        if len(possible_actions) == 0:
            return None
        
        actions_values = []
        ennemy_id = 3 - player_id
        for action in possible_actions:
            next_state = self.make_action(self.game_state, action, player_id)
            actions_values.append(-self.search(next_state, ennemy_id, depth))

        actions = [(action, actions_values[i]) for i, action in enumerate(possible_actions)]
        np.random.shuffle(actions)

        alpha_max = -float("inf")
        best_action = 0

        for action, alpha in actions:
            if alpha >= alpha_max:
                alpha_max = alpha
                best_action = action
        
        return best_action
    
    def search(self, game_state, player_id, depth):
        possible_actions = self.get_possible_actions(game_state)
        possible_states = []
        for action in possible_actions:
            next_state = self.make_action(game_state, action, player_id)
            possible_states.append(next_state)
        
        if depth == 0 or len(possible_states) == 0 or self.game_is_finished(game_state)[0]:
            return self.heuristic_value(game_state)
        
        ennemy_id = 3 - player_id
        alpha = max([-self.search(next_state, ennemy_id, depth-1) for next_state in possible_states])
        return alpha
    
    def make_action(self, game_state, actions, player_id):
        state_matrix = self.state_to_matrix(game_state)
        for action in actions:
            (x_s, y_s), number, (x_t, y_t) = action
            state_matrix[y_s, x_s, player_id] -= number
            state_matrix[y_t, x_t, player_id] += number
        new_game_state = self.matrix_to_state(state_matrix)
        
        for coords in new_game_state["commands"]:
            n_h, n_v, n_w = new_game_state["commands"][coords]
            
            # Battle handling
            if n_h * n_v != 0 or n_h * n_w != 0 or n_v * n_w != 0:

                # Battles H vs V (resp. W) and Q(H) < Q(V (resp. W))
                if (n_v * n_h != 0 and n_h <= n_v) or (n_w * n_h != 0 and n_h <= n_w):
                    if n_v != 0: n_v += n_h
                    else: n_w += n_h
                    n_h = 0
                
                # Battle V vs W and (Q(V) >= 1.5 Q(W) or Q(W) >= 1.5 Q(V))
                elif n_w * n_v != 0 and (n_v >= 1.5 * n_w or n_w >= 1.5 * n_v):
                    if n_v == max(n_v, n_w): n_w = 0
                    if n_w == max(n_v, n_w): n_v = 0
                
                # Random battle
                else:

                    # H vs V (resp. W)
                    if n_h * n_v != 0 or n_h * n_w != 0:
                        n_a = max(n_v, n_w)
                        p = 0.5 * n_a / n_h
                        if np.random.random() < p:
                            attacker_survivors = np.random.binomial(n_a, p)
                            humans_survivors = np.random.binomial(n_h, p)
                            if n_v != 0: n_v = attacker_survivors + humans_survivors
                            else: n_w = attacker_survivors + humans_survivors
                            n_h = 0
                        else:
                            humans_survivors = np.random.binomial(n_h, 1 - p)
                            if n_v != 0: n_v = 0
                            else: n_w = 0
                            n_h = humans_survivors
                    
                    # V vs W
                    elif n_v * n_w != 0:
                        p = 0.5 * n_v / n_w if n_v <= n_w else 1. * n_v / n_w - 0.5
                        if np.random.random() < p:
                            n_v = np.random.binomial(n_v, p)
                            n_w = 0
                        else:
                            n_v = 0
                            n_w = np.random.binomial(n_w, p)

            new_game_state["commands"][coords] = (n_h, n_v, n_w)
        
        return new_game_state

    def game_is_finished(self, game_state):
        player_id = self.player_id
        ennemy_id = 3 - self.player_id
        player_population = 1. * self.get_total_population(game_state, filter_id=player_id)
        ennemy_population = 1. * self.get_total_population(game_state, filter_id=ennemy_id)
        if player_population == 0:
            return True, -1
        if ennemy_population == 0:
            return True, 1
        return False, 0
