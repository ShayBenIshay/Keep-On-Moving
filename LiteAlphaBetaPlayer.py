from AlphaBetaPlayer import AlphaBetaPlayer


class LiteAlphaBetaPlayer(AlphaBetaPlayer):
    def __init__(self):
        AlphaBetaPlayer.__init__(self)

    # implement the Simple_Player heuristic
    def heuristic_function(self, board, loc):
        steps_available = 0
        for d in self.directions:
            i = loc[0] + d[0]
            j = loc[1] + d[1]
            if 0 <= i < len(board) and 0 <= j < len(board[0]) and board[i][j] == 0:  # then move is legal
                steps_available += 1

        if steps_available == 0:
            return -1, 0
        else:
            return 4 - steps_available, 0
