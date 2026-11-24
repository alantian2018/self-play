"""Compatibility entry point; prefer python -m experiments.connect4.train."""
from experiments.connect4.agent import Agent
from selfplay.checkpoints import load_checkpoint, save_checkpoint
from selfplay.config import Args
from selfplay.evaluation import evaluate
from selfplay.ppo import make_batch, update
from selfplay.train import main, train


if __name__ == "__main__":
    main()
