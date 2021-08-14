import time
import numpy as np


####
# for the debug
from MapsGenerator import ai_board
import matplotlib.pyplot as plt
####


def unavailable_steps(num_steps_available):
    if num_steps_available == 0:
        return -3
    else:
        return 4 - num_steps_available


def count_ones(board):
    counter = 0
    for i, row in enumerate(board):
        for j, val in enumerate(row):
            if val == 1.0:
                counter += 1
    return counter


# returns an estimation to the next_iteration_time in rb_minimax algorithm with a specific depth
def next_iteration_time_estimation(leaves, last_iteration_time):
    # return last_iteration_time * 4.5 * leaves
    return (1 + (6*leaves)/(3*leaves - 1)) * last_iteration_time
 # return (1 + 2*leaves)*last_iteration_time


class MinimaxHeuristic2s:
    def __init__(self):
        self.n = 0
        self.m = 0
        self.w = 0.95
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

    # initiates the player's location, and the rival's location
    def set_game_params(self, board):
        self.board = board
        self.n = len(self.board[0])  # cols number
        self.m = len(self.board)     # rows number

        self.width_radius_env = min(self.w_radius_env, int(self.n/2))
        self.length_radius_env = min(self.w_radius_env, int(self.m/2))

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
    #def debug_set_player_loc_backtracking(board, loc, width_radius, length_radius):
    def set_player_loc_backtracking(self, player, old_loc):
        if player==1:
            player_loc = self.loc
        else:
            player_loc = self.rival_loc
        debug_set_player_loc_backtracking(self.board, player_loc, 5, 5)
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
        my_unavailable_steps = unavailable_steps(steps_available)

        player_reachable_squares = self.reachable_white_squares(board, 1)
        rival_reachable_squares = self.reachable_white_squares(board, 2)

        reachable_diff = player_reachable_squares - rival_reachable_squares
        normalized_diff = reachable_diff / (self.m * self.n)
        # # if max val still very low give reachable diff extra adventage
        # if (player_reachable_squares+abs(reachable_diff))/(2*self.m*self.n) < 1-self.w:
        if (abs(reachable_diff))/(2*self.m*self.n) < 1-self.w:
        # if abs(normalized_diff) < 1-self.w:
        #     print(f"refactoring w:::: old w : {self.w} new w : {(self.w+1)/2} ")
        #     print(f"normalized diff = {normalized_diff} (1-w) = {1-self.w} other factor = { player_reachable_squares/(self.m*self.n)}  ")
        #     print(f"other factor would take this change? {(player_reachable_squares+reachable_diff)/(2*self.m*self.n) < 1-self.w}")
            self.w = (self.w*9+1)/10
        # else:
        #     print("HEY HEY, else gives a fight")
        normalized_unavailable = my_unavailable_steps / 3  # normalized to [-1,1]
        normalized_h_val = self.w * normalized_diff + (1 - self.w) * normalized_unavailable
        debug_heuristic(player_reachable_squares, rival_reachable_squares, reachable_diff, my_unavailable_steps,
                        normalized_h_val)
        return normalized_h_val

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

        # self.set_player_loc_backtracking(1,father_loc)
        return best_move, best_leaves, optimal_val

    def make_move(self, time_limit):
        # time_limit = time_limit*1.25
        start = time.time()

        # execute iterative deepening, and decide when to stop by approximating the next iteration time
        d = 5  # the initial depth limit
        move, leaves, optimal_val = self.return_best_move(self.board, d)

        last_iteration_time = time.time() - start
        next_iteration_time = next_iteration_time_estimation(leaves, last_iteration_time)
        time_until_now = time.time() - start

        while time_until_now + next_iteration_time < time_limit and not optimal_val:
            d += 1
            iteration_start_time = time.time()
            move, leaves, optimal_val = self.return_best_move(self.board, d)
            # print(f"move to: {move}")
            last_iteration_time = time.time() - iteration_start_time
            next_iteration_time = next_iteration_time_estimation(leaves, last_iteration_time)
            time_until_now = time.time() - start
            # print(f"time_until_now{time_until_now}, continue = {time_until_now + next_iteration_time < time_limit}")
            # print(f"next iteration prediction: {next_iteration_time}\n\n")

        # executes the best move
        debug_make_move(d,optimal_val)
        self.board[self.loc] = -1

        if move is None:
            exit()
        best_new_loc = (self.loc[0] + move[0], self.loc[1] + move[1])
        self.board[best_new_loc] = 1
        self.loc = best_new_loc

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
                return -1, 1, True
            elif is_player_stuck and is_rival_stuck:
                return 0, 1, True

        if agent_to_move == 2:
            if is_rival_stuck and (not is_player_stuck):
                return 1, 1, True
            elif is_rival_stuck and is_player_stuck:
                return 0, 1, True
        ############### end of goal state check ###############
        ########################################################

        if d == 0:
            #DOUBLE CHECK THIS: (shay)
            return self.heuristic_function(board_state, self.loc), 1, False

        if agent_to_move == 1:  # maximizer
            cur_max = float('-inf')
            cur_optimal = None
            total_leaves_num = 0

            assert count_ones(board_state) == 1
            father_loc = self.loc

            for child_state in (self.succ(board_state, 1)):
                v, leaves, optimal = self.rb_minimax_algorithm(child_state[0], 2, d - 1)
                # return to father's board_state
                # print("maximizing")
                self.set_player_loc_backtracking(1, father_loc)  # set the player back to his old position
                assert count_ones(board_state) == 1

                total_leaves_num += leaves
                cur_max = max(v, cur_max)
                if cur_max == v:
                    cur_optimal = optimal

            # print(f"maximizer child val {cur_max}")
            return cur_max, total_leaves_num, cur_optimal

        else:  # minimizer
            cur_min = float('inf')
            total_leaves_num = 0
            cur_optimal = None
            father_loc = self.rival_loc

            for child_state in (self.succ(board_state, 2)):
                v, leaves, optimal = self.rb_minimax_algorithm(child_state[0], 1, d - 1)
                # return to father's board_state
                # print("minimizing")
                self.set_player_loc_backtracking(2, father_loc)  # set the player back to his old position
                assert count_ones(board_state) == 1

                total_leaves_num += leaves
                cur_min = min(v, cur_min)
                if cur_min == v:
                    cur_optimal = optimal

            # print(f"minimizer child val {cur_min}")
            return cur_min, total_leaves_num, cur_optimal


##################################################
########## Functions for debugging ###############
##################################################
#switc the commented function and defined function for switching debug on/off
# def debug_heuristic(player_reachable_squares, rival_reachable_squares, reachable_diff, my_unavailable_steps,
#                     normalized_h_val):
#     print(f"player reachable squares {player_reachable_squares} rival reachable squares {rival_reachable_squares} ")
#     print(f"reachable diff is: {reachable_diff} , unavailable steps {my_unavailable_steps}")
#     print(f"normalized_h_val {normalized_h_val}")


def debug_make_move(d, optimal_val):
    print(f"Heuristic2 depth = {d}")
    print(f"Utility: {optimal_val}")
    print("**********************************************MOVE**********************************************\n\n\n")


# def debug_return_best_move():
#     print("###########################################")
#     print("Initial board : playing Minimax_rb_algorithm")
#
#
# def debug_set_player_loc_backtracking(board, loc, width_radius, length_radius):
#     print("the board before setting back:")
#     row_low_limit = max(0, loc[0] - length_radius)
#     row_high_limit = min(len(board) - 1, loc[0] + length_radius)
#     col_low_limit = max(0, loc[1] - width_radius)
#     col_high_limit = min(len(board[0]) - 1, loc[1] + width_radius)
#     mini_board = [row[col_low_limit:col_high_limit+1] for row in board[row_low_limit:row_high_limit+1]]
#     numpy_mini_board = np.array(mini_board)
#     print(numpy_mini_board)



def debug_heuristic(player_reachable_squares, rival_reachable_squares, reachable_diff, my_unavailable_steps, normalized_h_val):
    pass


# def debug_make_move(d,optimal_val):
#     pass


def debug_return_best_move():
    pass


def debug_set_player_loc_backtracking(board, loc, width_radius, length_radius):
    pass

# times = [1,2,3,4,5,6,7,8,9,10]
# # depths = []
#
# max_path_heuristic_results = []
# reachable_heuristic_results = []
# for time_limit in times:
#     reachable_agent = Reachable()
#     max_path = MaxPath()
#
#
#
#
#
# for t in np.linspace(0.1, 3, 50):
#     minimax = MinimaxPlayer()
#     minimax.set_game_params(ai_board.copy())
#     d = minimax.make_move(t)
#     times.append(t)
#     depths.append(d)
#     plt.scatter(times, depths)
#     plt.show()
