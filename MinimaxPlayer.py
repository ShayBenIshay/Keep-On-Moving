import time
import numpy as np


def unavailable_steps(num_steps_available):
    if num_steps_available == 0:
        return -1
    else:
        return 4 - num_steps_available


def count_ones(board):
    counter = 0
    for i, row in enumerate(board):
        for j, val in enumerate(row):
            if val == 1.0:
                counter += 1
    return counter


class MinimaxPlayer:
    def __init__(self):
        self.n = 0
        self.m = 0
        self.w = 0.3
        self.grey_w = 0.35

        # used to compute heuristic value:
        # initiates the limits of the self environment for the grey_counter squares
        self.width_radius_env = 0
        self.length_radius_env = 0
        self.w_radius_env = 7  # set the minimum radius for self env

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

        self.width_radius_env = min(self.w_radius_env, int(self.n/2))
        self.length_radius_env = min(self.w_radius_env, int(self.m/2))

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

    # Counts grey squares in the game's board
    def count_grey(self, board, agent_to_move):
        counter = 0
        player_loc = self.get_player_loc(agent_to_move)

        row_low_limit = max(0, player_loc[0] - self.length_radius_env)
        row_high_limit = min(self.m - 1, player_loc[0] + self.length_radius_env)
        col_low_limit = max(0, player_loc[1] - self.width_radius_env)
        col_high_limit = min(self.n - 1, player_loc[1] + self.width_radius_env)

        for i in range(row_low_limit, row_high_limit + 1):
            for j in range(col_low_limit, col_high_limit + 1):
                if board[i][j] == -1:
                    counter += 1
        return counter

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

            for d in self.directions:
                i = player_loc[0] + d[0]
                j = player_loc[1] + d[1]

                if row_low_limit <= i <= row_high_limit and col_low_limit <= j <= col_high_limit and board[i][
                    j] == 0:  # then move is legal
                    new_loc = (i, j)
                    # the square in new_loc is available in the game's board
                    if board[i][j] == 0 and (not reachable_board[i][j]):
                        reachable_board[i][j] = 1
                        reachable_squares.append(new_loc)
                        len_reachable_squares += 1
            start_index += 1
        # Returns the len of reachable_squares list.
        # This value presents the total number of reachable squares from the current player location in the game's board
        return len_reachable_squares - 1

    # returns an estimation to the next_iteration_time in rb_minimax algorithm with a specific depth
    def next_iteration_time_estimation(self, leaves, last_iteration_time):
        return (1 + (6 * leaves) / (3 * leaves - 1)) * last_iteration_time

    # Returns the state heuristic value, and the number of leaves which were developed during the function
    def heuristic_function(self, board, loc):
        steps_available = 0
        for d in self.directions:
            i = loc[0] + d[0]
            j = loc[1] + d[1]
            if 0 <= i < len(board) and 0 <= j < len(board[0]) and board[i][j] == 0:  # then move is legal
                steps_available += 1
        my_unavailable_steps = unavailable_steps(steps_available)

        ######### rival location env #############
        rival_loc = self.rival_loc
        ######### computing env dimension #########
        row_low_limit = max(0, rival_loc[0] - self.length_radius_env)
        row_high_limit = min(self.m - 1, rival_loc[0] + self.length_radius_env)
        col_low_limit = max(0, rival_loc[1] - self.width_radius_env)
        col_high_limit = min(self.n - 1, rival_loc[1] + self.width_radius_env)
        env_size = (row_high_limit - row_low_limit + 1) * (col_high_limit - col_low_limit + 1)
        #######################################
        # compute the heuristic values range to normalize
        grey_counter = self.count_grey(board, 1)

        player_reachable_squares = self.reachable_white_squares(board, 1)
        rival_reachable_squares = self.reachable_white_squares(board, 2)

        reachable_diff = player_reachable_squares - rival_reachable_squares
        #####################################

        # compute the range of the diff_reachable_value
        if grey_counter <= self.grey_w * (self.width_radius_env * self.length_radius_env):
            h_val = 0.3 * my_unavailable_steps + reachable_diff
        else:
            h_val = 0.1 * my_unavailable_steps + reachable_diff

        return h_val

    # returns list of successor states to the given board_state
    def succ(self, board_state, agent_to_move):
        player_loc = self.get_player_loc(agent_to_move)
        try:
            for direction in self.directions:
                # player_loc = self.get_player_loc(board_state, agent_to_move)
                i = player_loc[0] + direction[0]
                j = player_loc[1] + direction[1]
                if 0 <= i < len(board_state) and 0 <= j < len(board_state[0]) and board_state[i][
                    j] == 0:  # then move is legal
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
        #assert count_ones(self.board) == 1
        player1_loc = self.loc
        for child_state in self.succ(board, 1):

            minimax_val, leaves, optimal_temp = self.rb_minimax_algorithm(child_state[0], 2, d - 1)
            # return player 1 to his old location (backtracking)
            self.set_player_loc_backtracking(1, player1_loc)
            #assert count_ones(board) == 1

            if minimax_val > best_minimax_val:
                optimal_val = optimal_temp
                best_minimax_val = minimax_val
                best_move = child_state[1]
                best_leaves = leaves
        return best_move, best_leaves, optimal_val

    def make_move(self, time_limit):
        time_limit *= 0.9
        start = time.time()

        # execute iterative deepening, and decide when to stop by approximating the next iteration time
        d = 1  # the initial depth limit
        move, leaves, optimal_val = self.return_best_move(self.board, d)

        last_iteration_time = time.time() - start
        next_iteration_time = self.next_iteration_time_estimation(leaves, last_iteration_time)
        time_until_now = time.time() - start

        while time_until_now + next_iteration_time < time_limit and not optimal_val:
            iteration_start_time = time.time()
            d += 1
            move, leaves, optimal_val = self.return_best_move(self.board, d)
            last_iteration_time = time.time() - iteration_start_time
            next_iteration_time = self.next_iteration_time_estimation(leaves, last_iteration_time)
            time_until_now = time.time() - start

        # print(f"depth = {d}")
        # executes the best move
        # print(f"Utility: {optimal_val}")
        # print("***********************MOVE***********************\n\n\n")

        # executes the best move
        self.board[self.loc] = -1

        if move is None:
            exit()
        best_new_loc = (self.loc[0] + move[0], self.loc[1] + move[1])
        self.board[best_new_loc] = 1
        self.loc = best_new_loc

        #return d
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
        #  h_val = self.heuristic_function(board_state, self.loc)
            #return h_val[0], h_val[1], False

        if agent_to_move == 1:  # maximizer
            cur_max = float('-inf')
            cur_optimal = None
            total_leaves_num = 0

            assert count_ones(board_state) == 1
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
                assert count_ones(board_state) == 1

                total_leaves_num += leaves
                cur_min = min(v, cur_min)
                if cur_min == v:
                    cur_optimal = optimal

            return cur_min, total_leaves_num, cur_optimal
