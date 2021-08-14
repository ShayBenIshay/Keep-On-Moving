from MinimaxPlayer import MinimaxPlayer
from ContestPlayer import ContestPlayer

class AlphaBetaPlayer(MinimaxPlayer):
    def __init__(self):
        MinimaxPlayer.__init__(self)

    # returns an estimation to the next_iteration_time in rb_minimax algorithm with a specific depth
    def next_iteration_time_estimation(self, leaves, last_iteration_time):
        leaves = leaves ** 2
        return (1 + (6 * leaves) / (3 * leaves - 1)) * last_iteration_time

    def rb_alpha_beta(self, board_state, agent_to_move, d, alpha, beta):
        ############################################
        # check whether board_state is a goal state
        is_player_stuck = self.stuck(board_state, 1)
        is_rival_stuck = self.stuck(board_state, 2)
        if agent_to_move == 1:
            if is_player_stuck and (not is_rival_stuck):
                return -10000000, 1, True
            elif is_player_stuck and is_rival_stuck:
                return 0, 1, True

        if agent_to_move == 2:
            if is_rival_stuck and (not is_player_stuck):
                return 10000000, 1, True
            elif is_rival_stuck and is_player_stuck:
                return 0, 1, True
        ############### end of goal state check ###############
        ########################################################

        if d == 0:
            return self.heuristic_function(board_state, self.loc), 1, False
            #return h_val[0], h_val[1], False

        if agent_to_move == 1:  # maximizer
            cur_max = float('-inf')
            cur_optimal = None
            total_leaves_num = 0

            father_loc = self.loc

            for child_state in (self.succ(board_state, 1)):
                v, leaves, optimal = self.rb_alpha_beta(child_state[0], 2, d - 1, alpha, beta)
                # return to father's board_state
                self.set_player_loc_backtracking(1, father_loc)  # set the player back to his old position

                total_leaves_num += leaves
                cur_max = max(v, cur_max)
                if cur_max == v:
                    cur_optimal = optimal

                alpha = max(cur_max, alpha)
                if cur_max >= beta:     # cut
                    cur_max = float('inf')
                    break

            return cur_max, total_leaves_num, cur_optimal

        else:  # minimizer
            cur_min = float('inf')
            total_leaves_num = 0
            cur_optimal = None
            father_loc = self.rival_loc

            for child_state in (self.succ(board_state, 2)):
                v, leaves, optimal = self.rb_alpha_beta(child_state[0], 1, d - 1, alpha, beta)
                # return to father's board_state
                self.set_player_loc_backtracking(2, father_loc)  # set the player back to his old position

                total_leaves_num += leaves
                cur_min = min(v, cur_min)
                if cur_min == v:
                    cur_optimal = optimal

                beta = min(cur_min, beta)
                # cut
                if cur_min <= alpha:
                    cur_min = float('-inf')
                    break

            return cur_min, total_leaves_num, cur_optimal

    def rb_minimax_algorithm(self, board_state, agent_to_move, d):
        return self.rb_alpha_beta(board_state, agent_to_move, d, float('-inf'), float('inf'))