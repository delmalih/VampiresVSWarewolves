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

class IbrahimPlayer(BasePlayer):
    def __init__(self, name, sock):
        BasePlayer.__init__(self, name, sock)

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

        tmp = time.time()
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
        
        print(best_value, time.time() - tmp)
        return best_action
    
    ######################
    # Heuristic function #
    ######################

    def heuristic_value(self, game_state, player_id):
        # IDs
        ennemy_id = 3 - player_id
        human_id = 0
        
        # Compute populations
        total_pop = self.get_total_population(game_state)
        player_pop_score = 1. * self.get_total_population(game_state, filter_id=player_id) / total_pop
        ennemy_pop_score = 1. * self.get_total_population(game_state, filter_id=ennemy_id) / total_pop
        # Game finish
        if player_pop_score == 0:
            return -10000
        if ennemy_pop_score == 0:
            return 10000
        population_score = player_pop_score - ennemy_pop_score

        # More complex features
        def compute_scores_maps(game_state, id1, id2, feasability, total_pop=total_pop):
            max_dist = 1. * (game_state.shape[0] + game_state.shape[1])
            scores_maps = []
            Ys_1, Xs_1 = np.where(game_state[:, :, id1] > 0)
            for i in range(len(Xs_1)):
                # Get coords and number
                x_1 = Xs_1[i]
                y_1 = Ys_1[i]
                n_1 = game_state[y_1, x_1, id1]
                # Init. scores
                scores_map = np.zeros((game_state.shape[0], game_state.shape[1]))
                Ys_2, Xs_2 = np.where(game_state[:, :, id2] > 0)
                for j in range(len(Xs_2)):
                    # Get coords and number
                    x_2 = Xs_2[j]
                    y_2 = Ys_2[j]
                    n_2 = game_state[y_2, x_2, id2]
                    # Compute features
                    distance = (1. - max(abs(x_1 - x_2), abs(y_1 - y_2)) / max_dist) ** 20
                    population = n_2 / n_1
                    feasible = feasability(n_1, n_2)
                    scores_map[y_2, x_2] = distance * feasible * population
                scores_maps.append(scores_map)
            return np.array(scores_maps)

        # Computing scores
        human_player_scores = compute_scores_maps(game_state, player_id, human_id, feasability=lambda n1, n2: 1 if n1 >= n2 else 0)
        human_player_scores = np.max(human_player_scores, axis=0)
        human_ennemy_scores = compute_scores_maps(game_state, ennemy_id, human_id, feasability=lambda n1, n2: 1 if n1 >= n2 else 0)
        human_ennemy_scores = np.max(human_ennemy_scores, axis=0)
        player_ennemy_scores = compute_scores_maps(game_state, player_id, ennemy_id, feasability=lambda n1, n2: 1 if n1 >= 1.5 * n2 else 0)
        player_ennemy_scores = np.max(player_ennemy_scores)
        ennemy_player_scores = compute_scores_maps(game_state, ennemy_id, player_id, feasability=lambda n1, n2: 1 if n1 >= 1.5 * n2 else 0)
        ennemy_player_scores = np.max(ennemy_player_scores)
        human_scores = np.max(human_player_scores - human_ennemy_scores)

        return population_score * 1000 + (player_ennemy_scores - ennemy_player_scores) * 10 + human_scores

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

    def get_possible_actions(self, game_state, player_id):
        # Get game limits
        H, W = game_state.shape[:2]
        ennemy_id = 3 - player_id

        # Populations
        n_player = np.sum(game_state[:, :, player_id])
        n_ennemy = np.sum(game_state[:, :, ennemy_id])
        threat = n_ennemy >= 1.5 * n_player

        # Get possible actions
        possible_actions = []
        Ys, Xs = np.where(game_state[:, :, player_id])
        for i in range(len(Xs)):
            x, y = Xs[i], Ys[i]
            number = int(game_state[y, x][player_id])
            if number:
                if y > 0:
                    if threat or np.sum(game_state[:y, :, :]) != 0:
                        possible_actions.append([((x, y), number, (x, y-1))])
                if y < H - 1:
                    if threat or np.sum(game_state[y+1:, :, :]) != 0:
                        possible_actions.append([((x, y), number, (x, y+1))])

                if x > 0:
                    if threat or np.sum(game_state[:, :x, :]) != 0:
                        possible_actions.append([((x, y), number, (x-1, y))])
                    if y > 0:
                        if threat or np.sum(game_state[:y, :x, :]) != 0:
                            possible_actions.append([((x, y), number, (x-1, y-1))])
                    if y < H - 1:
                        if threat or np.sum(game_state[y+1:, :x, :]) != 0:
                            possible_actions.append([((x, y), number, (x-1, y+1))])
                
                if x < W - 1:
                    if threat or np.sum(game_state[:, x+1:, :]) != 0:
                        possible_actions.append([((x, y), number, (x+1, y))])
                    if y > 0:
                        if threat or np.sum(game_state[:y, x+1:, :]) != 0:
                            possible_actions.append([((x, y), number, (x+1, y-1))])
                    if y < H - 1:
                        if threat or np.sum(game_state[y+1:, x+1:, :]) != 0:
                            possible_actions.append([((x, y), number, (x+1, y+1))])

        return possible_actions
    
    def hash_state(self, game_state, player_id):
        return hash(str(game_state) + str(player_id))