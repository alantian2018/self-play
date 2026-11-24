from dataclasses import dataclass

from selfplay.config import Args as PPOArgs


@dataclass
class Args(PPOArgs):
    experiment: str = "connect4"
    wandb_project: str = "self-play-connect4"
    checkpoint_dir: str = "checkpoints/connect4"
    num_iterations: int = 100_000
    games_per_update: int = 50
    num_envs: int = 50
    save_every: int = 10_000
    eval_every: int = 1000
    eval_games: int = 64
    eval_num_envs: int = 32
    max_opponents: int = 500
    winrate_threshold: float = 0.75
    min_opponent_games: int = 400
    learning_rate: float = 2.5e-4
    minibatch_size: int = 64
    update_epochs: int = 4
    gamma: float = 0.99
