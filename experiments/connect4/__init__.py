from pettingzoo.classic import connect_four_v3

from .agent import Agent
from .config import Args

player_labels = ("red (first)", "black (second)")


def make_env(render_mode=None):
    return connect_four_v3.env(render_mode=render_mode, screen_scaling=4)


def make_agent():
    return Agent()


def default_args():
    return Args()
