from copy import deepcopy

import numpy as np
import torch.nn as nn

from .rollout import Rollout, copy_observation


def random_policy(obs):
    return int(np.random.choice(np.flatnonzero(obs["action_mask"])))


class SelfPlay:
    def __init__(self, alpha=0.02):
        self.opponents = []
        self.winrates = []
        self.opponent_games = []
        self.random_winrates = np.array([0.5, 0.5])
        self.alpha = alpha

    def play_a_game(self, env, current_policy, opponent_policy, opponent_idx=None, *, frames=None, starting_idx=None):
        """Play one game, optionally recording rendered frames."""
        game = self._game(env, current_policy, opponent_policy, opponent_idx, frames, starting_idx)
        try:
            policy, obs = next(game)
            while True:
                policy, obs = game.send(policy(obs))
        except StopIteration as finished:
            return finished.value

    def play_games(self, envs, current_policy, num_games, *, records=None):
        """Advance several games together, batching turns that use the same policy."""
        if not envs:
            raise ValueError("At least one environment is required")
        histories = []
        for start in range(0, num_games, len(envs)):
            active = []
            game_records = {}
            for env in envs[:min(len(envs), num_games - start)]:
                starting_idx = np.random.randint(2)
                opponent, idx = self.sample_opponent(starting_idx)
                trace = [] if records is not None else None
                game = self._game(env, current_policy, opponent, idx,
                                  starting_idx=starting_idx, action_trace=trace)
                if records is not None:
                    game_records[game] = {"opponent": idx, "starting_idx": starting_idx, "actions": trace}
                policy, obs = next(game)
                active.append((game, policy, obs))

            while active:
                groups = {}
                for game, policy, obs in active:
                    groups.setdefault(policy, []).append((game, obs))
                active = []
                for policy, turns in groups.items():
                    observations = [obs for _, obs in turns]
                    if hasattr(policy, "act_batch"):
                        actions = policy.act_batch(observations)
                    else:
                        actions = [policy(obs) for obs in observations]
                    for (game, _), action in zip(turns, actions):
                        try:
                            next_policy, obs = game.send(int(action))
                            active.append((game, next_policy, obs))
                        except StopIteration as finished:
                            histories.append(finished.value)
                            if records is not None:
                                records.append(game_records.pop(game))
        return histories

    def _game(self, env, current_policy, opponent_policy, opponent_idx, frames=None, starting_idx=None, action_trace=None):
        """Pause before each action so the caller can batch policy inference."""
        env.reset()
        if frames is not None:
            frames.append(env.render())
        first, second = env.agents
        if starting_idx is None:
            starting_idx = np.random.randint(2)  # Current policy: 0=first, 1=second.
        if starting_idx:
            first, second = second, first

        policies = {first: current_policy, second: opponent_policy}
        histories = {first: Rollout(), second: Rollout()}
        pending = {}

        for agent in env.agent_iter():
            obs, reward, terminated, truncated, _ = env.last()

            # Finish this player's previous move after the opponent has responded.
            if agent in pending:
                previous_obs, previous_action = pending.pop(agent)
                histories[agent].add_step(
                    previous_obs, previous_action, reward, obs, terminated, truncated,
                )

            if terminated or truncated:
                env.step(None)
                continue

            saved_obs = copy_observation(obs)
            action = yield policies[agent], obs
            pending[agent] = (saved_obs, action)
            env.step(action)
            if action_trace is not None:
                action_trace.append(action)
            if frames is not None:
                frames.append(env.render())

        if opponent_idx is not None or opponent_policy is random_policy:
            # Current policy's score: win=1, draw=0.5, loss=0.
            score = (sum(histories[first].reward) + 1) / 2
            wr = self.random_winrates if opponent_idx is None else self.winrates[opponent_idx]
            wr[starting_idx] += self.alpha * (score - wr[starting_idx])
            if opponent_idx is not None:
                self.opponent_games[opponent_idx][starting_idx] += 1

        return histories[first], histories[second]

    def add_opponent(self, opp):
        opp = deepcopy(opp)
        if isinstance(opp, nn.Module):
            opp.eval()
            opp.requires_grad_(False)
        self.opponents.append(opp)
        self.winrates.append(np.array([0.5, 0.5]))  # Going first, going second.
        self.opponent_games.append(np.zeros(2, dtype=int))

    def sample_opponent(self, starting_idx):
        """Prefer opponents that are harder for the current policy in this seat."""
        if not self.opponents or np.random.rand() < 0.05:
            return random_policy, None
        weights = 1 - np.array(self.winrates)[:, starting_idx]
        if weights.sum() == 0:
            weights = np.ones(len(self.opponents))
        idx = int(np.random.choice(len(self.opponents), p=weights / weights.sum()))
        return self.opponents[idx], idx
