import numpy as np
import torch
import wandb

from experiments import get_experiment
from .self_play import SelfPlay


def game_video(experiment, record, reward):
    """Replay the actual sampled game's actions; don't sample a replacement game."""
    env = experiment.make_env(render_mode="rgb_array")
    try:
        env.reset()
        frames = [env.render()]
        for action in record['actions']:
            env.step(action)
            frames.append(env.render())
        frames.extend([frames[-1]] * 4)
        idx = record['opponent']
        opponent = f"opponent_{idx}" if idx is not None else 'random'
        label = experiment.player_labels[record['starting_idx']]
        return wandb.Video(
            np.stack(frames).transpose(0, 3, 1, 2), fps=2, format="mp4",
            caption=f"Current: {label}; vs {opponent}; reward={reward:+g}; sampled evaluation game",
        )
    finally:
        env.close()


def evaluate(agent, self_play, experiment=None, num_games=64, num_envs=32):
    """Evaluate seat-aware matchups in batches, with a loss-preferred replay."""
    if num_games < 1 or num_envs < 1:
        raise ValueError("eval_games and eval_num_envs must be at least 1")
    experiment = experiment or get_experiment('connect4')
    numpy_state = np.random.get_state()
    was_training = agent.training
    # Use training's sampling scores, held fixed throughout evaluation.
    collector = SelfPlay(alpha=0)
    collector.opponents = self_play.opponents
    collector.winrates = [wr.copy() for wr in self_play.winrates]
    collector.opponent_games = [np.zeros(2, dtype=int) for _ in self_play.opponents]
    envs = []
    try:
        with torch.random.fork_rng(devices=[]):
            agent.eval()
            envs = [experiment.make_env() for _ in range(min(num_envs, num_games))]
            records = []
            pairs = collector.play_games(envs, agent, num_games, records=records)
            rewards = np.array([sum(current.reward) for current, _ in pairs])
            losses = np.flatnonzero(rewards < 0)
            selected = int(np.random.choice(losses if losses.size else np.arange(num_games)))
            record = records[selected]
            metrics = {
                'eval/mean_reward': rewards.mean(),
                'eval/video': game_video(experiment, record, rewards[selected]),
            }
            seats = np.array([r['starting_idx'] for r in records])
            metrics['eval/second_player_fraction'] = seats.mean()
            for seat, name in enumerate(('first', 'second')):
                scores = rewards[seats == seat]
                if len(scores):
                    metrics[f'eval/win_rate_{name}'] = (scores > 0).mean()
            return metrics
    finally:
        for env in envs:
            env.close()
        agent.train(was_training)
        np.random.set_state(numpy_state)
