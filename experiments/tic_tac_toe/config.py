from dataclasses import dataclass

from selfplay.config import Args as PPOArgs


@dataclass
class Args(PPOArgs):
    experiment: str = "tic_tac_toe"
    wandb_project: str = "self-play-tic-tac-toe"
    checkpoint_dir: str = "checkpoints/tic_tac_toe"
    num_iterations: int = 200_000
    games_per_update: int = 200
    num_envs: int = 200
    save_every: int = 10_000
    snapshot_every: int | None = 2500
    eval_every: int = 1000
    eval_games: int = 1000
    eval_num_envs: int = 100
    winrate_threshold: float = 0.51
    min_opponent_games: int = 2000
    learning_rate: float = 2.5e-4
    minibatch_size: int = 64
    update_epochs: int = 4
    gamma: float = 1.0
