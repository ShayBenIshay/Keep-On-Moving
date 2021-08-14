from AlphaBetaPlayer import AlphaBetaPlayer


class HeavyAlphaBetaPlayer(AlphaBetaPlayer):
    def __init__(self):
        AlphaBetaPlayer.__init__(self)

    # return tuple x:
    # x[0]: the length of max path from current board_state
    # x[1]: the number of reachable squares founded at the end of the max_len path founded
    # x[2]: the number of leaves were developed in this function
    def max_path_len(self, agent, depth):
        if depth == 0:
            # return self.reachable_white_squares(self.board, agent)
            return 0, self.reachable_white_squares(self.board, agent), 1

        row_low_limit = 0
        row_high_limit = self.m - 1
        col_low_limit = 0
        col_high_limit = self.n - 1
        reach_final = 0
        father_loc = self.get_player_loc(agent)
        max_reach_from_current_loc = 0

        leaves_counter = 0
        for d in self.directions:
            i = father_loc[0] + d[0]
            j = father_loc[1] + d[1]

            # then move is legal
            if row_low_limit <= i <= row_high_limit and col_low_limit <= j <= col_high_limit and self.board[i][
                j] == 0:
                new_loc = (i, j)
                self.loc = new_loc
                self.board[father_loc] = -1
                self.board[new_loc] = agent
                reach_for_this_path, reach_at_end_state, leaves_num = self.max_path_len(agent, depth - 1)

                leaves_counter += leaves_num

                reach_for_this_path += 1
                if max_reach_from_current_loc < reach_for_this_path:
                    reach_final = reach_at_end_state
                    max_reach_from_current_loc = reach_for_this_path
                self.set_player_loc_backtracking(agent, father_loc)

        return max_reach_from_current_loc, reach_final, leaves_counter

    # Returns the state heuristic value, and the number of leaves which were developed during the function
    def heuristic_function(self, board, loc):
        max_len = self.max_path_len(1, 4)
        h_val = max_len[1] - (self.reachable_white_squares(board, 2) - max_len[0])
        h_norm = self.m * self.n
        return h_val / h_norm
