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
python play_connect4.py --checkpoint path/to/connect4.pt
python play_tic_tac_toe.py --checkpoint path/to/tic_tac_toe.pt
```

Omit `--checkpoint` to load the newest checkpoint, or use **Load checkpoint** in the window. Choose who goes first and click to play.

## Train

```bash
wandb login
python -m experiments.connect4.train
python -m experiments.tic_tac_toe.train
```

Each game has its own `config.py` and `agent.py` under `experiments/`. Checkpoints save to `checkpoints/<game>/<run-datetime>/`.

Resume with a total iteration target greater than the saved iteration:

```bash
python -m experiments.tic_tac_toe.train --checkpoint path/to/checkpoint.pt --num-iterations 200000
```

## W&B runs

- [Tic-tac-toe](https://forge.coreweave.com/wandb/alantian2018/self-play-tic-tac-toe/overview)
- [Connect 4](https://forge.coreweave.com/wandb/alantian2018/self-play-connect4?nw=nwuseralantian2018)
