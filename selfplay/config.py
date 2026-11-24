from dataclasses import dataclass


@dataclass
class Args:
    experiment: str = "connect4"
    wandb_project: str = "self-play"
    checkpoint: str | None = None
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10_000
    seed: int | None = None
    num_iterations: int = 100_000
    games_per_update: int = 100
    num_envs: int = 64
    eval_every: int = 50  # Set to 0 to disable evaluation videos.
    eval_games: int = 64
    eval_num_envs: int = 32
    max_opponents: int = 100
    winrate_threshold: float = 0.7
    min_opponent_games: int = 200
    learning_rate: float = 2.5e-4
    minibatch_size: int = 64
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
