import time
import numpy as np


def unavailable_steps(num_steps_available):
    if num_steps_available == 0:
        return -3
    else:
        return 4 - num_steps_available


class TempContestPlayer:
    def __init__(self):
        self.n = 0
        self.m = 0
        self.rival_loc = None
        self.loc = None
        self.board = None
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.max_heuristic_value = 0
        self.min_heuristic_value = 0

    # initiates the player's location, and the rival's location
    def set_game_params(self, board):
        self.board = board
        self.n = len(self.board[0])  # cols number
        self.m = len(self.board)     # rows number
        self.max_heuristic_value = self.m * self.n + 1  # max reachable squares and +1 for unavailable steps
        self.min_heuristic_value = -self.max_heuristic_value
        for i, row in enumerate(board):
            for j, val in enumerate(row):
                if val == 1:
                    self.loc = (i, j)
                    break

        for i, row in enumerate(board):
            for j, val in enumerate(row):
                if val == 2:
                    self.rival_loc = (i, j)
                    break

    # moving a player back from the backtracking process
    # player is the agent
    # old loc is location to move back
    # set self.<player/rival> to old location on board
    def set_player_loc_backtracking(self, player, old_loc):
        if player == 1:
            self.board[self.loc] = 0
            self.loc = old_loc
        else:
            self.board[self.rival_loc] = 0
            self.rival_loc = old_loc

        self.board[old_loc] = player

    # gets the location of specific player
    def get_player_loc(self, player):
        assert(player == 1 or player == 2)
        if player == 1:
            return self.loc
        return self.rival_loc

    # Returns the total number of reachable squares from the current player location in the game's board
    # player's value is 1 or 2
    def reachable_white_squares(self, board, player):
        # initiates boolean board. True if the square is reachable, and False if not
        initial_player_loc = self.get_player_loc(player)

        row_low_limit = 0
        row_high_limit = self.m - 1
        col_low_limit = 0
        col_high_limit = self.n - 1

        reachable_board = np.zeros((self.m, self.n))
        reachable_squares = [initial_player_loc]
        start_index = 0
        len_reachable_squares = 1

        # adds reachable squares to reachable_squares list
        while start_index < len_reachable_squares:
            player_loc = reachable_squares[start_index]
            count_moves_available = 0
            for d in self.directions:
                i = player_loc[0] + d[0]
                j = player_loc[1] + d[1]
                if row_low_limit <= i <= row_high_limit and col_low_limit <= j <= col_high_limit and board[i][
                    j] == 0:  # then move is legal
                    new_loc = (i, j)
                    count_moves_available += 1
                    # the square in new_loc is available in the game's board
                    if board[i][j] == 0 and (not reachable_board[i][j]):
                        reachable_board[i][j] = 1
                        reachable_squares.append(new_loc)
                        len_reachable_squares += 1
            start_index += 1
        # Returns the len of reachable_squares list.
        # This value presents the total number of reachable squares from the current player location in the game's board
        return len_reachable_squares - 1

    # Returns the state heuristic value
    def heuristic_function(self, board, loc):
        steps_available = 0
        for d in self.directions:
            i = loc[0] + d[0]
            j = loc[1] + d[1]
            if 0 <= i < len(board) and 0 <= j < len(board[0]) and board[i][j] == 0:  # then move is legal
                steps_available += 1
        my_unavailable_steps = 0.001*(unavailable_steps(steps_available)/3) # values in range [0.5,-0.5]
        player_reachable_squares = self.reachable_white_squares(board, 1)
        rival_reachable_squares = self.reachable_white_squares(board, 2)

        reachable_diff = player_reachable_squares - rival_reachable_squares
        h_val = my_unavailable_steps + reachable_diff
        return h_val

    # returns list of successor states to the given board_state
    def succ(self, board_state, agent_to_move):
        player_loc = self.get_player_loc(agent_to_move)
        try:
            for direction in self.directions:
                i = player_loc[0] + direction[0]
                j = player_loc[1] + direction[1]
                if 0 <= i < len(board_state) and 0 <= j < len(board_state[0]) and board_state[i][j] == 0:  # then move is legal
                    new_loc = (i, j)

                    board_state[player_loc] = -1
                    board_state[new_loc] = agent_to_move
                    if agent_to_move == 1:
                        self.loc = new_loc
                    else:
                        self.rival_loc = new_loc
                    yield board_state, direction
        except StopIteration:
            return None

    def return_best_move(self, board, d):
        best_minimax_val = float('-inf')
        best_move = None
        best_leaves = 0
        optimal_val = None
        # saving player 1 loc for backtracking
        player1_loc = self.loc
        for child_state in self.succ(board, 1):

            minimax_val, leaves, optimal_temp = self.rb_minimax_algorithm(child_state[0], 2, d - 1)
            # return player 1 to his old location (backtracking)
            self.set_player_loc_backtracking(1, player1_loc)

            if minimax_val > best_minimax_val:
                optimal_val = optimal_temp
                best_minimax_val = minimax_val
                best_move = child_state[1]
                best_leaves = leaves

        return best_move, best_leaves, optimal_val

    def make_move(self, time_limit):
        time_limit = time_limit*0.95
        start = time.time()

        # execute iterative deepening, and decide when to stop by approximating the next iteration time
        d = 4  # the initial depth limit
        move, leaves, optimal_val = self.return_best_move(self.board, d)

        last_iteration_time = time.time() - start
        next_iteration_time = self.next_iteration_time_estimation(leaves, last_iteration_time)
        time_until_now = time.time() - start

        while time_until_now + next_iteration_time < time_limit and not optimal_val:
            d += 1
            iteration_start_time = time.time()
            move, leaves, optimal_val = self.return_best_move(self.board, d)
            last_iteration_time = time.time() - iteration_start_time
            next_iteration_time = self.next_iteration_time_estimation(leaves, last_iteration_time)
            time_until_now = time.time() - start

        # executes the best move
        self.board[self.loc] = -1

        if move is None:
            exit()
        best_new_loc = (self.loc[0] + move[0], self.loc[1] + move[1])
        self.board[best_new_loc] = 1
        self.loc = best_new_loc
        # return d
        return move

    # sets the new rival's location
    def set_rival_move(self, loc):
        self.board[self.rival_loc] = -1
        self.rival_loc = loc
        self.board[loc] = 2

    # returns True if agent_to_move is stuck. else, returns False
    def stuck(self, board_state, agent_to_move):
        for direction in self.directions:
            player_loc = self.get_player_loc(agent_to_move)
            i = player_loc[0] + direction[0]
            j = player_loc[1] + direction[1]
            if 0 <= i < len(board_state) and 0 <= j < len(board_state[0]) and board_state[i][
                j] == 0:  # then move is legal
                return False
        return True

    # returns tuple which contains the minimax value of board_state, and the number of expanded leaves in this iteration
    def rb_minimax_algorithm(self, board_state, agent_to_move, d):
        ############################################
        # check whether board_state is a goal state
        is_player_stuck = self.stuck(board_state, 1)
        is_rival_stuck = self.stuck(board_state, 2)
        if agent_to_move == 1:
            if is_player_stuck and (not is_rival_stuck):
                return self.min_heuristic_value, 1, True
            elif is_player_stuck and is_rival_stuck:
                return 0, 1, True

        if agent_to_move == 2:
            if is_rival_stuck and (not is_player_stuck):
                return self.max_heuristic_value, 1, True
            elif is_rival_stuck and is_player_stuck:
                return 0, 1, True
        ############### end of goal state check ###############
        ########################################################

        if d == 0:
            return self.heuristic_function(board_state, self.loc), 1, False
            # h_val = self.heuristic_function(board_state, self.loc)
            # return h_val[0], 1 + h_val[1], False

        if agent_to_move == 1:  # maximizer
            cur_max = float('-inf')
            cur_optimal = None
            total_leaves_num = 0

            father_loc = self.loc

            for child_state in (self.succ(board_state, 1)):
                v, leaves, optimal = self.rb_minimax_algorithm(child_state[0], 2, d - 1)
                # return to father's board_state
                self.set_player_loc_backtracking(1, father_loc)  # set the player back to his old position

                total_leaves_num += leaves
                cur_max = max(v, cur_max)
                if cur_max == v:
                    cur_optimal = optimal

            return cur_max, total_leaves_num, cur_optimal

        else:  # minimizer
            cur_min = float('inf')
            total_leaves_num = 0
            cur_optimal = None
            father_loc = self.rival_loc

            for child_state in (self.succ(board_state, 2)):
                v, leaves, optimal = self.rb_minimax_algorithm(child_state[0], 1, d - 1)
                # return to father's board_state
                self.set_player_loc_backtracking(2, father_loc)  # set the player back to his old position

                total_leaves_num += leaves
                cur_min = min(v, cur_min)
                if cur_min == v:
                    cur_optimal = optimal

            return cur_min, total_leaves_num, cur_optimal

    # returns an estimation to the next_iteration_time in rb_minimax algorithm with a specific depth
    def next_iteration_time_estimation(self, leaves, last_iteration_time):
        return (1 + (6 * leaves) / (3 * leaves - 1)) * last_iteration_time


class ContestPlayer(TempContestPlayer):
    def __init__(self):
        TempContestPlayer.__init__(self)

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
                return self.min_heuristic_value, 1, True
            elif is_player_stuck and is_rival_stuck:
                return 0, 1, True

        if agent_to_move == 2:
            if is_rival_stuck and (not is_player_stuck):
                return self.max_heuristic_value, 1, True
            elif is_rival_stuck and is_player_stuck:
                return 0, 1, True
        ############### end of goal state check ###############
        ########################################################

        if d == 0:
            return self.heuristic_function(board_state, self.loc), 1, False

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

    def make_move(self, time_limit):
        return super().make_move(time_limit)