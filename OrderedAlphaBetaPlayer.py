from AlphaBetaPlayer import AlphaBetaPlayer
import time


# order only root children
class OrderedAlphaBetaPlayer(AlphaBetaPlayer):
    def __init__(self):
        AlphaBetaPlayer.__init__(self)

    # returns an estimation to the next_iteration_time in rb_minimax algorithm with a specific depth
    def next_iteration_time_estimation(self, leaves, last_iteration_time):
        leaves = leaves ** 2
        return (1 + (6 * leaves) / (3 * leaves - 1)) * last_iteration_time

    # expand successor states in a given order, according to ordered_directions list
    def ordered_succ(self, board_state, agent_to_move, ordered_directions):
        player_loc = self.get_player_loc(agent_to_move)
        try:
            for direction in ordered_directions:
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

    def return_best_move_ordered(self, board, d, children_order):
        # children_order = tuples of form (direction, minimax_val)
        if len(children_order) == 0:
            children = self.succ(board, 1)
        else:
            ordered_directions = [child[0] for child in children_order]
            children = self.ordered_succ(board, 1, ordered_directions)

        best_minimax_val = float('-inf')
        best_move = None
        best_leaves = 0
        optimal_val = None
        new_children_order = []

        father_loc = self.get_player_loc(1)
        for c in children:
            minimax_val, leaves, optimal_temp = self.rb_alpha_beta(c[0], 2, d, float('-inf'), float('inf'))
            #print(f"minimax_val {minimax_val}")
            # return player 1 to his old location (backtracking)
            self.set_player_loc_backtracking(1, father_loc)

            new_children_order.append((c[1], minimax_val))
            if minimax_val > best_minimax_val:
                optimal_val = optimal_temp
                best_minimax_val = minimax_val
                best_move = c[1]
                best_leaves = leaves
        #print("\n\n")
        return best_move, best_leaves, optimal_val, new_children_order

    def make_move(self, time_limit):
        time_limit *= 0.9
        start = time.time()

        # execute iterative deepening, and decide when to stop by approximating the next iteration time
        d = 1   # the initial depth limit

        move, leaves, optimal_val, children_minimax_vals = self.return_best_move_ordered(self.board, d, [])
        last_iteration_time = time.time() - start
        next_iteration_time = self.next_iteration_time_estimation(leaves, last_iteration_time)
        time_until_now = time.time() - start

        while time_until_now + next_iteration_time < time_limit and optimal_val == 0:
            d += 1
            iteration_start_time = time.time()

            # print(f"original: {children_minimax_vals}")
            # sort the root's children in ascending order, to get a better pruning of the tree
            ordered_children_lst = sorted(children_minimax_vals, key=lambda child: child[1], reverse=True)
            # print(f"order: {ordered_children_lst}\n\n")

            move, leaves, optimal_val, children_minimax_vals = self.return_best_move_ordered(self.board, d, ordered_children_lst)

            last_iteration_tme = time.time() - iteration_start_time
            next_iteration_time = self.next_iteration_time_estimation(leaves, last_iteration_time)
            time_until_now = time.time() - start

        #print(f"Ordered depth = {d}")
        # executes the best move
       # print(f"Utility: {optimal_val}")
        #print("***********************MOVE***********************\n\n\n")
        self.board[self.loc] = -1

        if move is None:
            exit()

        best_new_loc = (self.loc[0] + move[0], self.loc[1] + move[1])
        self.board[best_new_loc] = 1

        self.loc = best_new_loc
        #return d
        return move



