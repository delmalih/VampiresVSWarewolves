# -*- coding: UTF-8 -*-

###########
# Imports #
###########

""" Global """
import sys
import numpy as np
from easydict import EasyDict as edict
from scipy.stats import multivariate_normal
import scipy.special
import time

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
        height, width = self.game_state.shape[:2]; k = 10
        gaussian_matrix = np.array([
            [
                np.sqrt(2 * np.pi * k) * multivariate_normal.pdf(max(abs(x - width), abs(y - height)), mean=0, cov=k)
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
            next_states, next_probabilities = self.make_action(self.game_state, action, self.player_id)

            # Handling game end
            if len(next_states) == 1:
                game_finish, winner = self.game_is_finished(next_states[0], self.player_id)
                if game_finish and winner == 1:
                    return action

            # Choose the best action
            action_value = sum([next_probabilities[i] * self.max_value(next_state, self.player_id, depth, alpha, beta) for i, next_state in enumerate(next_states)])
            if action_value > best_value:
                best_value = action_value
                best_action = action
        
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

        # Game end
        if player_pop_value == 0:
            return -10000
        elif ennemy_pop_value == 0:
            return 10000

        # More complex features
        sign_func_human = lambda x: np.where(x >= 0, 1, 0)
        sign_func_ennemy = lambda x: np.where(x >= 0, 1, -1)

        # human - player distance
        human_player_dist_val = 0
        Ys, Xs = np.where(game_state[:, :, player_id] > 0)
        for i in range(len(Xs)):
            x = Xs[i]; y = Ys[i]
            human_player_diff = np.where(game_state[:, :, 0] == 0, 0, sign_func_human(game_state[y, x, player_id] - game_state[:, :, 0]))
            human_player_dist_val += np.max(human_player_diff * self.gaussian_weights[(x, y)] * (game_state[:, :, 0] / total_pop))
        
        # human - ennemy distance
        human_ennemy_dist_val = 0
        Ys, Xs = np.where(game_state[:, :, ennemy_id] > 0)
        for i in range(len(Xs)):
            x = Xs[i]; y = Ys[i]
            human_ennemy_diff = np.where(game_state[:, :, 0] == 0, 0, sign_func_human(game_state[y, x, ennemy_id] - game_state[:, :, 0]))
            human_ennemy_dist_val += np.max(human_ennemy_diff * self.gaussian_weights[(x, y)] * (game_state[:, :, 0] / total_pop))
        
        # player - ennemy distance
        player_ennemy_dist_val = 0
        Ys, Xs = np.where(game_state[:, :, player_id] > 0)
        for i in range(len(Xs)):
            x = Xs[i]; y = Ys[i]
            player_ennemy_diff = np.where(game_state[:, :, ennemy_id] == 0, 0, sign_func_ennemy(game_state[y, x, player_id] - 1.5 * game_state[:, :, ennemy_id]))
            player_ennemy_dist_val += np.max(player_ennemy_diff * self.gaussian_weights[(x, y)] * (game_state[:, :, ennemy_id] / total_pop))

        return pop_value * 1000 + player_ennemy_dist_val * 10 + (human_player_dist_val - human_ennemy_dist_val)

    #############
    # Algorithm #
    #############
    
    def max_value(self, game_state, player_id, depth, alpha, beta):
        # Get actions
        possible_actions = self.get_possible_actions(game_state, player_id)

        # Game finish
        game_finish, winner = self.game_is_finished(game_state, player_id)
        if game_finish:
            return winner * 100000
        if depth <= 0 or len(possible_actions) == 0:
            return self.heuristic_value(game_state, player_id)

        # Compute max value
        value = -float("inf")
        for action in possible_actions:
            next_states, next_probabilities = self.make_action(game_state, action, player_id)
            value = max(value, sum([next_probabilities[i] * self.min_value(next_state, player_id, depth-1, alpha, beta) for i, next_state in enumerate(next_states)]))
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
            return winner * 10000
        if depth <= 0 or len(possible_actions) == 0:
            return self.heuristic_value(game_state, player_id)

        # Compute max value
        value = float("inf")
        for action in possible_actions:
            next_states, next_probabilities = self.make_action(game_state, action, player_id)
            value = min(value, sum([next_probabilities[i] * self.max_value(next_state, player_id, depth-1, alpha, beta) for i, next_state in enumerate(next_states)]))
            if value <= alpha:
                return value
            beta = min(beta, value)
        
        return value
    
    ###################
    # Utils functions #
    ###################

    def make_action(self, game_state, actions, player_id):
        all_final_states = []
        all_probabilities = []
        new_game_state = game_state.copy()
        for action in actions:
            (x_s, y_s), number, (x_t, y_t) = action
            new_game_state[y_s, x_s, player_id] -= number
            new_game_state[y_t, x_t, player_id] += number
            
            n_h, n_v, n_w = new_game_state[y_t, x_t]
            n_player = n_v if player_id == 1 else n_w
            n_ennemy = n_w if player_id == 1 else n_v
            ennemy_id = 3 - player_id

            # Player vs Ennemy battle
            if n_player * n_ennemy != 0:
                if n_player >= 1.5 * n_ennemy:
                    new_game_state[y_t, x_t, ennemy_id] = 0
                    all_final_states.append(new_game_state)
                    all_probabilities.append(1.0)
                elif n_player <= 1.5 * n_ennemy:
                    new_game_state[y_t, x_t, player_id] = 0
                    all_final_states.append(new_game_state)
                    all_probabilities.append(1.0)
                else:
                    # Probability
                    p = 0.5 * n_player / n_ennemy if n_player < n_ennemy else 1. * n_player / n_ennemy - 0.5
                    # Player loses
                    current_state = new_game_state.copy()
                    current_state[y_t, x_t, player_id] = 0
                    all_final_states.append(current_state)
                    all_probabilities.append(1.0 - p + p * (1. - p) ** n_player)
                    # Player wins
                    current_state = new_game_state.copy()
                    current_state[y_t, x_t, player_id] = n_player * p
                    all_final_states.append(current_state)
                    all_probabilities.append(p * (1. - (1. - p) ** n_player))

            # Human vs Player (or Ennemy) battle
            elif n_h * n_player != 0 or n_h * n_ennemy != 0:
                if (n_h * n_player != 0 and n_h <= n_player) or (n_h * n_ennemy != 0 and n_h <= n_ennemy):
                    new_game_state[y_t, x_t, 0] = 0
                    new_game_state[y_t, x_t, player_id] = n_player + n_h if n_player != 0 else n_player
                    new_game_state[y_t, x_t, ennemy_id] = n_ennemy + n_h if n_ennemy != 0 else n_ennemy
                    all_final_states.append(new_game_state)
                    all_probabilities.append(1.0)
                else:
                    # Probability
                    creature_id = player_id if n_player != 0 else ennemy_id
                    n_creature = max(n_player, n_ennemy)
                    p = 0.5 * n_creature / n_h
                    # Creature loses
                    current_state = new_game_state.copy()
                    current_state[y_t, x_t, creature_id] = 0
                    all_final_states.append(current_state)
                    all_probabilities.append(1.0 - p + p * (1. - p) ** (n_creature + n_h))
                    # Creature wins
                    current_state = new_game_state.copy()
                    current_state[y_t, x_t, creature_id] = (n_creature + n_h) * p
                    all_final_states.append(current_state)
                    all_probabilities.append(p * (1. - (1. - p) ** (n_creature + n_h)))
            
            # No battle
            else:
                all_final_states.append(new_game_state)
                all_probabilities.append(1.0)
        
        return all_final_states, all_probabilities

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
