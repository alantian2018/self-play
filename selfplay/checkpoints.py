from copy import deepcopy

import numpy as np
import torch


def save_checkpoint(path, agent, optimizer, self_play, iteration, last_snapshot, games):
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_rng = np.random.get_state()
    torch.save({
        "agent": agent.state_dict(),
        "policy_class": f"{type(agent).__module__}.{type(agent).__qualname__}",
        "optimizer": optimizer.state_dict(),
        "opponents": [opp.state_dict() for opp in self_play.opponents],
        "opponent_ids": self_play.opponent_ids,
        "next_opponent_id": self_play.next_opponent_id,
        "winrates": [wr.tolist() for wr in self_play.winrates],
        "opponent_games": [counts.tolist() for counts in self_play.opponent_games],
        "random_winrates": self_play.random_winrates.tolist(),
        "alpha": self_play.alpha,
        "iteration": iteration,
        "last_snapshot": last_snapshot,
        "games": games,
        "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
        "torch_rng": torch.get_rng_state(),
    }, path)


def load_checkpoint(path, agent, optimizer, self_play, make_agent=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    weights = state.get("agent", state)
    policy_class = f"{type(agent).__module__}.{type(agent).__qualname__}"
    if state.get("policy_class", policy_class) != policy_class:
        raise ValueError("Checkpoint policy does not match the selected experiment's policy class.")
    try:
        agent.load_state_dict(weights)
    except RuntimeError as error:
        raise ValueError("Checkpoint architecture does not match the selected policy.") from error
    if "agent" not in state:  # Older checkpoints contain only model weights.
        return {}
    optimizer.load_state_dict(state["optimizer"])
    self_play.opponents = []
    for weights in state["opponents"]:
        opponent = make_agent() if make_agent is not None else deepcopy(agent)
        opponent.load_state_dict(weights)
        self_play.opponents.append(opponent.eval().requires_grad_(False))
    self_play.opponent_ids = list(state.get("opponent_ids", range(len(self_play.opponents))))
    self_play.next_opponent_id = state.get("next_opponent_id", max(self_play.opponent_ids, default=-1) + 1)
    self_play.winrates = [np.array(wr) for wr in state["winrates"]]
    self_play.opponent_games = [
        np.array(counts, dtype=int)
        for counts in state.get("opponent_games", [[0, 0] for _ in self_play.opponents])
    ]
    self_play.random_winrates = np.array(state["random_winrates"])
    self_play.alpha = state["alpha"]
    self_play.trim_opponents()
    return state
