"""An optimal opponent used only for evaluation, never for PPO training."""
from functools import cache

import numpy as np
from pettingzoo.classic import tictactoe_v3

from selfplay.self_play import SelfPlay

LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
         (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))


def next_board(board, action):
    # Place our mark, then swap perspectives for the opponent's turn.
    return tuple(-1 if i == action else -mark for i, mark in enumerate(board))


@cache
def value(board):
    """Perfect-play outcome from the player-to-move's perspective."""
    for line in LINES:
        total = sum(board[i] for i in line)
        if abs(total) == 3:
            return total // 3
    legal = [i for i, mark in enumerate(board) if mark == 0]
    return max((-value(next_board(board, action)) for action in legal), default=0)


def minimax_policy(obs):
    planes = obs["observation"]
    board = tuple((planes[..., 0] - planes[..., 1]).ravel().tolist())
    legal = np.flatnonzero(obs["action_mask"])
    scores = np.array([-value(next_board(board, action)) for action in legal])
    return int(np.random.choice(legal[scores == scores.max()]))


def evaluation_metrics(agent, games_per_seat=50):
    """Sample both starting seats against random tie-breaking perfect play."""
    env = tictactoe_v3.env()
    collector = SelfPlay()
    metrics = {}
    try:
        for seat, name in enumerate(("first", "second")):
            rewards = []
            for _ in range(games_per_seat):
                current, _ = collector.play_a_game(env, agent, minimax_policy, starting_idx=seat)
                rewards.append(sum(current.reward))
            rewards = np.array(rewards)
            for outcome, mask in (("win", rewards > 0), ("draw", rewards == 0), ("loss", rewards < 0)):
                metrics[f"eval/minimax_{outcome}_rate_{name}"] = mask.mean()
        metrics["eval/minimax_loss_rate"] = (
            metrics["eval/minimax_loss_rate_first"] + metrics["eval/minimax_loss_rate_second"]
        ) / 2
    finally:
        env.close()
    return metrics
