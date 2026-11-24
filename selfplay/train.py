import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
from tqdm import trange

from experiments import get_experiment
from .checkpoints import load_checkpoint, save_checkpoint
from .evaluation import evaluate
from .ppo import update
from .self_play import SelfPlay


def train(args):
    if args.save_every < 1:
        raise ValueError("save_every must be at least 1")
    if args.num_envs < 1 or args.games_per_update < 1:
        raise ValueError("num_envs and games_per_update must be at least 1")
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    experiment = get_experiment(args.experiment)
    agent = experiment.make_agent()
    self_play = SelfPlay()
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    state = load_checkpoint(args.checkpoint, agent, optimizer, self_play, experiment.make_agent) if args.checkpoint else {}
    if not self_play.opponents:
        self_play.add_opponent(agent)
    start_iteration = state.get("iteration", 0)
    last_snapshot = state.get("last_snapshot", 0)
    games = state.get("games", 0)
    if start_iteration >= args.num_iterations:
        raise ValueError(f"Checkpoint is at iteration {start_iteration}; increase --num-iterations")
    run_datetime = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S-%f")
    with wandb.init(project=args.wandb_project, group=args.experiment, config=vars(args)) as run:
        if state:
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"])
        envs = [experiment.make_env() for _ in range(min(args.num_envs, args.games_per_update))]
        try:
            progress = trange(start_iteration, args.num_iterations, initial=start_iteration,
                              total=args.num_iterations, desc="Training")
            for iteration in progress:
                pairs = self_play.play_games(envs, agent, args.games_per_update)
                rollouts = [current for current, _ in pairs]
                metrics = update(agent, optimizer, rollouts, args)
                if (
                    (all(wr.mean() > 0.6 for wr in self_play.winrates)
                     and all(counts.sum() > 2000 for counts in self_play.opponent_games))
                    or iteration + 1 - last_snapshot >= 0.1 * args.num_iterations
                ):
                    self_play.add_opponent(agent)
                    last_snapshot = iteration + 1
                rewards = np.array([sum(rollout.reward) for rollout in rollouts])
                games += args.games_per_update
                mean_winrates = np.mean(self_play.winrates, axis=1)
                lowest = int(np.argmin(mean_winrates))
                metrics.update({
                    "train/mean_reward": rewards.mean(),
                    "train/win_rate": (rewards > 0).mean(),
                    "train/draw_rate": (rewards == 0).mean(),
                    "train/second_player_fraction": np.mean([
                        current.obs[0]["observation"].sum() > 0 for current in rollouts
                    ]),
                    "train/games": games,
                    "train/lowest_winrate": mean_winrates[lowest],
                    "train/lowest_winrate_opponent": lowest,
                    "self_play/opponents": len(self_play.opponents),
                })
                winrates = {"random": self_play.random_winrates}
                winrates.update({f"opponent_{i}": wr for i, wr in enumerate(self_play.winrates)})
                for name, wr in winrates.items():
                    metrics[f"winrate/{name}"] = wr.mean()
                    metrics[f"winrate_first/{name}"] = wr[0]
                    metrics[f"winrate_second/{name}"] = wr[1]
                for i, counts in enumerate(self_play.opponent_games):
                    metrics[f"games/opponent_{i}"] = counts.sum()
                if args.eval_every and (
                    (iteration + 1) % args.eval_every == 0 or iteration + 1 == args.num_iterations
                ):
                    metrics.update(evaluate(agent, self_play, experiment, args.eval_games, args.eval_num_envs))
                run.log(metrics, step=iteration + 1)
                progress.set_postfix(mean_reward=f"{rewards.mean():.3f}", opps=len(self_play.opponents))
                if (iteration + 1) % args.save_every == 0 or iteration + 1 == args.num_iterations:
                    path = Path(args.checkpoint_dir) / run_datetime / f"iteration_{iteration + 1}.pt"
                    save_checkpoint(path, agent, optimizer, self_play, iteration + 1, last_snapshot, games)
                    run.summary["checkpoint"] = str(path)
        finally:
            for env in envs:
                env.close()
    return agent


def main(default_experiment="connect4"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("connect4", "tic_tac_toe"), default=default_experiment)
    for name in ("checkpoint", "checkpoint_dir", "save_every", "num_iterations", "num_envs",
                 "games_per_update", "eval_every", "eval_games", "eval_num_envs", "seed"):
        kind = str if name in ("checkpoint", "checkpoint_dir") else int
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=argparse.SUPPRESS)
    options = vars(parser.parse_args())
    args = get_experiment(options["experiment"]).default_args()
    for name, value in options.items():
        setattr(args, name, value)
    train(args)
