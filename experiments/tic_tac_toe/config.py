from dataclasses import dataclass

from selfplay.config import Args as PPOArgs


@dataclass
class Args(PPOArgs):
    experiment: str = "tic_tac_toe"
    wandb_project: str = "self-play-tic-tac-toe"
    checkpoint_dir: str = "checkpoints/tic_tac_toe"
    num_iterations: int = 200000
    games_per_update: int = 100
    num_envs: int = 64
    save_every: int = 20000
    eval_every: int = 1000
    eval_games: int = 64
    eval_num_envs: int = 32
    learning_rate: float = 2.5e-4
    minibatch_size: int = 64
    update_epochs: int = 4
    gamma: float = 1.0
