from pettingzoo.classic import tictactoe_v3

from .agent import Agent
from .config import Args

player_labels = ("X (first)", "O (second)")


def make_env(render_mode=None):
    return tictactoe_v3.env(render_mode=render_mode, screen_height=400)


def make_agent():
    return Agent()


def default_args():
    return Args()
