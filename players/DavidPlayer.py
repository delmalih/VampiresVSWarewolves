# -*- coding: UTF-8 -*-

###########
# Imports #
###########

""" Global """
import sys
import numpy as np
from easydict import EasyDict as edict
from scipy.stats import multivariate_normal

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
        height, width = self.game_state.shape[:2]; k = 5
        gaussian_matrix = np.array([
            [
                2 * np.pi * k * multivariate_normal.pdf([x - width, y - height], mean=np.zeros((2)), cov=k*np.eye(2))
                for x in range(2 * width)
            ] for y in range(2 * height)
        ])
        self.gaussian_weights = {}
        for y in range(height):
            for x in range(width):
                self.gaussian_weights[(x, y)] = gaussian_matrix[height - y:2*height - y, width - x:2*width - x]

    def get_next_actions(self, depth=4):
        possible_actions = self.get_possible_actions(self.game_state, self.player_id)
        if len(possible_actions) == 0:
            return None
        
        # Shuffle actions
        np.random.shuffle(possible_actions)
        
        # Init. variables
        best_action = None
        best_value = -float("inf")
        alpha = -float("inf")
        beta = float("inf")

        # Looping over all possible actions
        for action in possible_actions:
            # Compute next state
            next_state = self.make_action(self.game_state, action, self.player_id)

            # Handling game end
            game_finish, winner = self.game_is_finished(next_state, self.player_id)
            if game_finish and winner == 1:
                return action

            action_value = self.max_value(next_state, self.player_id, depth, alpha, beta)
            if action_value > best_value:
                best_value = action_value
                best_action = action
        
        print(best_value, best_action)
        return best_action
    
    ######################
    # Heuristic function #
    ######################

    def heuristic_value(self, game_state, player_id):
        # IDs
        ennemy_id = 3 - player_id
        
        # Compute populations
        total_pop = self.get_total_population(game_state)
        player_pop_value = 1. * self.get_total_population(game_state, filter_id=player_id) / total_pop
        ennemy_pop_value = 1. * self.get_total_population(game_state, filter_id=ennemy_id) / total_pop
        pop_value = player_pop_value - ennemy_pop_value

        # More complex features
        sign_func = lambda x: np.where(x >= 0, 1, 0)

        # human - player distance
        human_player_dist_val = 0
        Ys, Xs = np.where(game_state[:, :, player_id] > 0)
        for i in range(len(Xs)):
            x = Xs[i]; y = Ys[i]
            human_player_diff = np.where(game_state[:, :, 0] == 0, 0, sign_func(game_state[y, x, player_id] - game_state[:, :, 0]))
            human_player_dist_val += np.sum(human_player_diff * self.gaussian_weights[(x, y)] * (1. * game_state[:, :, 0]) / total_pop)
        
        # human - ennemy distance
        human_ennemy_dist_val = 0
        Ys, Xs = np.where(game_state[:, :, ennemy_id] > 0)
        for i in range(len(Xs)):
            x = Xs[i]; y = Ys[i]
            human_ennemy_diff = np.where(game_state[:, :, 0] == 0, 0, sign_func(game_state[y, x, ennemy_id] - game_state[:, :, 0]))
            human_ennemy_dist_val += np.sum(human_ennemy_diff * self.gaussian_weights[(x, y)] * (1. * game_state[:, :, 0]) / total_pop)
        
        # player - ennemy distance
        player_ennemy_dist_val = 0
        Ys, Xs = np.where(game_state[:, :, player_id] > 0)
        for i in range(len(Xs)):
            x = Xs[i]; y = Ys[i]
            player_ennemy_diff = np.where(game_state[:, :, ennemy_id] == 0, 0, sign_func(game_state[y, x, player_id] - 1.5*game_state[:, :, 0]))
            player_ennemy_dist_val += np.sum(player_ennemy_diff * self.gaussian_weights[(x, y)] * (1. * game_state[:, :, ennemy_id]) / total_pop)

        return pop_value * 1000 + player_ennemy_dist_val * 10 + human_player_dist_val - human_ennemy_dist_val

    #############
    # Algorithm #
    #############
    
    def max_value(self, game_state, player_id, depth, alpha, beta):
        # Get actions
        possible_actions = self.get_possible_actions(game_state, player_id)

        # Game finish
        game_finish, winner = self.game_is_finished(game_state, player_id)
        if game_finish:
            return winner * float("inf")
        if depth == 0 or len(possible_actions) == 0:
            return self.heuristic_value(game_state, player_id)

        # Compute max value
        value = -float("inf")
        for action in possible_actions:
            next_state = self.make_action(game_state, action, player_id)
            value = max(value, self.min_value(next_state, player_id, depth-1, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        
        return value
    
    def min_value(self, game_state, player_id, depth, alpha, beta):
        # Get actions
        possible_actions = self.get_possible_actions(game_state, player_id)

        # Game finish
        game_finish, winner = self.game_is_finished(game_state, player_id)
        if game_finish:
            return winner * float("inf")
        if depth == 0 or len(possible_actions) == 0:
            return self.heuristic_value(game_state, player_id)

        # Compute max value
        value = float("inf")
        for action in possible_actions:
            next_state = self.make_action(game_state, action, player_id)
            value = min(value, self.max_value(next_state, player_id, depth-1, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)
        
        return value
    
    ###################
    # Utils functions #
    ###################

    def make_action(self, game_state, actions, player_id):
        new_game_state = game_state.copy()
        for action in actions:
            (x_s, y_s), number, (x_t, y_t) = action
            new_game_state[y_s, x_s, player_id] -= number
            new_game_state[y_t, x_t, player_id] += number
            
            n_h, n_v, n_w = new_game_state[y_t, x_t]

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
                        p = (0.5 * n_a / n_h) ** 2
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
                        p = (0.5 * n_v / n_w if n_v <= n_w else 1. * n_v / n_w - 0.5) ** 10
                        if np.random.random() < p:
                            n_v = np.random.binomial(n_v, p)
                            n_w = 0
                        else:
                            n_v = 0
                            n_w = np.random.binomial(n_w, p)

            new_game_state[y_t, x_t, 0] = n_h
            new_game_state[y_t, x_t, 1] = n_v
            new_game_state[y_t, x_t, 2] = n_w
        
        return new_game_state

    def game_is_finished(self, game_state, player_id):
        ennemy_id = 3 - player_id
        player_population = 1. * self.get_total_population(game_state, filter_id=player_id)
        ennemy_population = 1. * self.get_total_population(game_state, filter_id=ennemy_id)
        if player_population == 0:
            return True, -1
        if ennemy_population == 0:
            return True, 1
        return False, 0

    def get_total_population(self, game_state, filter_id=None):
        total = 0
        if filter_id is not None:
            total += np.sum(game_state[:, :, filter_id])
        else:
            total += np.sum(game_state)
        return total
