# PPO Self-Play

Batched PPO self-play for Connect 4 and tic-tac-toe, with frozen opponents and W&B logging.

## Install

Python 3.10+ with Tk for the game windows. From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Play a checkpoint

```bash
python play_connect4.py
python play_tic_tac_toe.py
```

Loads a model from `release_checkpoints/<game>/` (falls back to local training saves). Use `--checkpoint path/to/model.pt` or **Load checkpoint** to choose another. Choose who goes first and click to play.

## Train

```bash
wandb login
python -m experiments.connect4.train
python -m experiments.tic_tac_toe.train
```

Each game has its own `config.py` and `agent.py` under `experiments/`. Full training checkpoints save to `checkpoints/<game>/<run-datetime>/` and are ignored by Git.

Resume with a total iteration target greater than the saved iteration:

```bash
python -m experiments.tic_tac_toe.train --checkpoint path/to/checkpoint.pt --num-iterations 200000
```

## Release a model

Export selected weights into the tracked `release_checkpoints/` directory:

```bash
python export_checkpoint.py checkpoints/tic_tac_toe/<run>/iteration_120000.pt release_checkpoints/tic_tac_toe/iteration_120000.pt
```

Releases omit optimizer state and frozen opponents. Use full training checkpoints to resume; releases can only warm-start a new run.


## W&B runs

- [Tic-tac-toe](https://forge.coreweave.com/wandb/alantian2018/self-play-tic-tac-toe/overview)
- [Connect 4](https://forge.coreweave.com/wandb/alantian2018/self-play-connect4?nw=nwuseralantian2018)
