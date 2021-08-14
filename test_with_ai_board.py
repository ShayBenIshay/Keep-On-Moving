from MapsGenerator import ai_board
import numpy as np
from MinimaxPlayer import MinimaxPlayer
from ContestPlayer import ContestPlayer
from AlphaBetaPlayer import AlphaBetaPlayer
from OrderedAlphaBetaPlayer import OrderedAlphaBetaPlayer
import matplotlib.pyplot as plt


def test_minimax():
    times = []
    depths = []

    for t in np.linspace(0.1, 3, 50):
        player = ContestPlayer()
        player.set_game_params(ai_board.copy())
        d = player.make_move(t)
        times.append(t)
        depths.append(d)

    plt.scatter(times, depths)
    plt.show()


def test_alpha_beta():
    times = []
    depths = []

    for t in np.linspace(0.1, 3, 50):
        player = AlphaBetaPlayer()
        player.set_game_params(ai_board.copy())
        d = player.make_move(t)
        times.append(t)
        depths.append(d)

    plt.scatter(times, depths)
    plt.show()


def test_ordered_alpha_beta():
    times = []
    depths = []

    for t in np.linspace(0.1, 3, 50):
        player = OrderedAlphaBetaPlayer()
        player.set_game_params(ai_board.copy())
        d = player.make_move(t)
        times.append(t)
        depths.append(d)

    plt.scatter(times, depths)
    plt.show()


def main():
    test_minimax()
    # test_alpha_beta()
    # test_ordered_alpha_beta()


if __name__ == '__main__':
    main()
