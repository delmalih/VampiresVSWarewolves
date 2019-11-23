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

    def heuristic_value(self, game_state, player_id):
        # Game finish
        game_finish, winner = self.game_is_finished(game_state, player_id)
        if game_finish:
            return winner * float("inf")

        # IDs
        ennemy_id = 3 - player_id

        # Compute all positions and populations
        human_positions = []; human_populations = []
        player_positions = []; player_populations = []
        ennemy_positions = []; ennemy_populations = []
        for coords in game_state["commands"]:
            if game_state["commands"][coords][0] > 0:
                human_positions.append(coords)
                human_populations.append(game_state["commands"][coords][0])
            if game_state["commands"][coords][player_id] > 0:
                player_positions.append(coords)
                player_populations.append(game_state["commands"][coords][player_id])
            if game_state["commands"][coords][ennemy_id] > 0:
                ennemy_positions.append(coords)
                ennemy_populations.append(game_state["commands"][coords][ennemy_id])
        
        # Compute populations
        total_pop = sum(human_populations) + sum(player_populations) + sum(ennemy_populations)
        player_pop_value = 1. * sum(player_populations) / total_pop
        ennemy_pop_value = 1. * sum(ennemy_populations) / total_pop
        pop_value = player_pop_value - ennemy_pop_value

        # human - player & human - ennemy distances
        def compute_dist_pop_value(positions1, populations1, positions2, populations2, dist_func_1, dist_func_2):
            distance_matrix = self.compute_distance_matrix(positions1, positions2, dist_func=dist_func_1)
            population_matrix = self.compute_distance_matrix(populations1, populations2, dist_func=dist_func_2)
            dist_pop_matrix = population_matrix + np.exp(-distance_matrix/10.)
            dist_pop_value = 0 if np.prod(dist_pop_matrix.shape) == 0 else np.max(dist_pop_matrix)
            return dist_pop_value
        
        human_player_value = compute_dist_pop_value(human_positions, human_populations,
                                                    player_positions, player_populations,
                                                    dist_func_1=lambda x1, x2: np.linalg.norm(x1 - x2, ord=1),
                                                    dist_func_2=lambda x1, x2: 1 if x1 == x2 else -np.sign(x1 - x2))
        
        human_ennemy_value = compute_dist_pop_value(human_positions, human_populations,
                                                    ennemy_positions, ennemy_populations,
                                                    dist_func_1=lambda x1, x2: np.linalg.norm(x1 - x2, ord=1),
                                                    dist_func_2=lambda x1, x2: 1 if x1 == x2 else -np.sign(x1 - x2))
        
        player_ennemy_value = compute_dist_pop_value(player_positions, player_populations,
                                                     ennemy_positions, ennemy_populations,
                                                     dist_func_1=lambda x1, x2: np.linalg.norm(x1 - x2, ord=1),
                                                     dist_func_2=lambda x1, x2: 1 if x1 == 1.5*x2 else -np.sign(x1 - 1.5*x2))
        
        dist_pop_value = (human_player_value - human_ennemy_value) + player_ennemy_value * 0.5

        return pop_value * 100 + dist_pop_value

    def compute_distance_matrix(self, positions1, positions2, dist_func=np.linalg.norm):
        distance_matrix = np.zeros((len(positions1), len(positions2)))
        for i, vect1 in enumerate(positions1):
            for j, vect2 in enumerate(positions2):
                distance_matrix[i, j] = dist_func(np.array(vect1), np.array(vect2))
        return distance_matrix

    #####################
    # MiniMax algorithm #
    #####################

    def best_move(self, player_id, depth=2):
        possible_actions = self.get_possible_actions(self.game_state, player_id)
        if len(possible_actions) == 0:
            return None
        
        ennemy_id = 3 - player_id
        best_action = None
        best_value = -float("inf")
        np.random.shuffle(possible_actions)
        for action in possible_actions:
            next_state = self.make_action(self.game_state, action, player_id)
            action_value = -self.search(next_state, ennemy_id, depth)
            if action_value > best_value:
                best_value = action_value
                best_action = action
        
        print(best_value, best_action)
        return best_action
    
    def search(self, game_state, player_id, depth):
        possible_actions = self.get_possible_actions(game_state, player_id)
        possible_states = []
        for action in possible_actions:
            next_state = self.make_action(game_state, action, player_id)
            possible_states.append(next_state)
        
        if depth == 0 or len(possible_states) == 0 or self.game_is_finished(game_state, player_id)[0]:
            return self.heuristic_value(game_state, player_id)
        
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

            new_game_state["commands"][coords] = (n_h, n_v, n_w)
        
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
        for coord in game_state["commands"]:
            if filter_id is not None:
                total += game_state["commands"][coord][filter_id]
            else:
                total += sum(game_state["commands"][coord])
        return total
