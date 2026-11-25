# Self-play experiments

Use the `rl` conda environment. Run these commands from the repository root:

```bash
conda run -n rl python -m experiments.connect4.train
conda run -n rl python -m experiments.tic_tac_toe.train
```

Each experiment owns its policy architecture in `agent.py` and training settings
in `config.py`. Its `__init__.py` defines the environment, `make_agent()` factory,
and player labels, and exposes that experiment's config through `default_args()`.
Edit either policy independently; checkpoints reconstruct frozen opponents using
that experiment's factory. Shared code lives in `selfplay/`:

- `policy.py`: optional masked sampling and PPO policy helpers, with no network architecture.
- `self_play.py` and `rollout.py`: opponent pool and batched game collection.
- `ppo.py`: advantage calculation and PPO updates.
- `train.py`, `evaluation.py`, and `checkpoints.py`: training, W&B videos/metrics, and resume.
- `config.py`: common defaults inherited by each experiment's own config.

Change `experiments/connect4/config.py` or `experiments/tic_tac_toe/config.py`
to tune one game independently. Any common setting can be overridden by declaring
it in that experiment's `Args` dataclass. Command-line overrides still take
precedence, and W&B records the resolved settings.

To bring your own architecture, replace the `Agent` class in an experiment's
`agent.py` and return it from `make_agent()`. It must be a PyTorch module with:

- `get_action_and_value(obs, masks, action=None)`: batched actions, log probabilities,
  entropy, and values. All outputs have shape `(batch,)`.
- `act_batch(observations)`: one legal integer action per observation dict.
- `forward(observation)`: one legal integer action for a single observation dict.

Training observations have shape `(batch, height, width, channels)`, and masks
have shape `(batch, actions)` with boolean entries. How you process them is up
to the policy. Inheriting `Policy` is optional; its helpers implement the three
methods above if you provide `get_logits(obs)` and
`get_logits_and_value(obs)`. The latter returns unmasked logits and scalar values
for each board. Keeping these two methods separate avoids critic work during play.
Neither shared training nor checkpoint code requires an encoder or actor/critic
attribute names. Architectures are editable independently, but changing tensor
shapes requires a fresh run or an explicit weight migration.

The Connect 4 policy enforces left–right reflection symmetry by averaging logits
from the board and its mirror, reversing the mirrored column logits. Value
predictions are averaged too. Both views run in one batch; this doubles neural
network work but shares learning between reflected positions. Gravity prevents
using vertical flips or rotations. Tic-tac-toe's policy is independent.
Existing Connect 4 weights still load, but this changes their predictions and
those of frozen opponents; start a fresh run to avoid stale saved win rates.

Tic-tac-toe averages all eight rotations/reflections in a single inference batch.
Action logits are mapped back to the original PettingZoo cell indices before
averaging; values are invariant. Occupied-cell masks are applied afterward in the
original coordinates. This adds eight-view network computation, not eight games.
Existing weights still load, but their predictions change. Older tic-tac-toe
checkpoints reset saved opponent scores and game counts on resume; newly saved
checkpoints retain them. Custom policies can bump their optional `policy_version`
when inference behavior changes without changing weight shapes.

Connect 4 retains the existing settings, including 5% random opponents,
seat-specific EMA scores with alpha 0.02, and promotion above 60% against every
frozen opponent after more than 200 games each. Configure these gates with
`winrate_threshold` and `min_opponent_games` in each experiment's config, or
`--winrate-threshold 0.6 --min-opponent-games 200`. Both comparisons are strict:
the scores and counts must exceed these values. A snapshot is also forced after
10% of the total target iterations without one. Tic-tac-toe uses gamma 1.0;
its training length and checkpoint interval are set in its own config.

Collection chooses the current policy's starting seat with equal probability,
then samples frozen opponents proportionally to `1 - winrate[opponent, seat]`.
The same selected seat is used to play the game and update its score. Promotion
and the main W&B win-rate curves still use the average of both seats.

`max_opponents` caps the frozen pool (default 20, also configurable with
`--max-opponents`). When adding a snapshot at capacity, the existing opponent
with the highest average score is evicted: the one the current policy beats most
easily. The new snapshot is retained. Random is not counted. Opponent IDs stay
stable after eviction so W&B curves continue to refer to the same frozen policy.
Resuming with a smaller cap trims the restored pool using its saved scores.

Set `snapshot_every` in either experiment config (or `--snapshot-every 1000`) to
force a new opponent after that many training iterations without a snapshot.
The default `None` keeps the 10%-of-`num_iterations` fallback. Win-rate promotion
can add a snapshot earlier and resets this timer. These are PPO iterations,
not individual moves or games; `save_every` separately controls checkpoint files.

Checkpoints go to `checkpoints/connect4/<run-datetime>/` or
`checkpoints/tic_tac_toe/<run-datetime>/`, with chronologically sortable local
timestamps such as `2026-11-24_15-30-00-123456`. W&B keeps its default run naming.
Resume with a total iteration target:

```bash
conda run -n rl python -m experiments.tic_tac_toe.train \
  --checkpoint checkpoints/tic_tac_toe/<run-datetime>/iteration_500.pt \
  --num-iterations 1000
```

Current Connect 4 CNN checkpoints still load in the Connect 4 experiment.
Weights cannot be shared between games with different board/action dimensions.
The root `ppo.py`, `self_play.py`, and `utils.py` remain small
compatibility entry points for existing commands/imports.

W&B uses separate projects: `self-play-connect4` and `self-play-tic-tac-toe`.
Set `wandb_project` in each experiment's `config.py` to change its destination.
New runs use these projects; existing runs remain in their original project.
Logging includes per-opponent training
scores, first/second-seat scores, game counts, videos, and PPO losses.
`train/lowest_winrate` is the lowest frozen-opponent score averaged over both
seats; `train/lowest_winrate_opponent` is its zero-based opponent number.
EMA scores count draws as half a win.
`train/second_player_fraction` and `eval/second_player_fraction` show the fraction
of sampled games where the current policy goes second (expected near 0.5).

At each evaluation interval, `eval_games` games (default 64) run across
`eval_num_envs` environments (default 32), using the same seat-aware weighted
sampling and 5% random opponents as training. Evaluation holds the training
sampling scores fixed and does not update the training pool. The only W&B eval
panels are `eval/win_rate_first`, `eval/win_rate_second`, `eval/mean_reward`,
`eval/second_player_fraction`, and `eval/video`. Win rates count wins only; draws are included in the denominator.
If a small evaluation batch has no games in one seat, that seat's metric is omitted.

`eval/video` replays one actual game chosen uniformly from this evaluation batch.
Its caption identifies the current policy's seat, opponent, and result. The
aggregate evaluation rates describe the whole batch. Only the
selected game is rendered. Configure these sizes in each experiment's config or
with `--eval-games` and `--eval-num-envs`.

Tic-tac-toe's optional minimax evaluator lives in
`experiments/tic_tac_toe/minimax.py`; it is available for separate checks and is
not logged automatically during training.

For a quick local smoke run without uploads or video encoding:

```bash
WANDB_MODE=disabled conda run -n rl python -m experiments.tic_tac_toe.train \
  --num-iterations 2 --games-per-update 16 --num-envs 8 --eval-every 0
```

Run the tests with `conda run -n rl python -m unittest discover -s tests`.
