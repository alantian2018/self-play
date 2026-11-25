from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from pettingzoo.classic.tictactoe.board import Board, TTT_GAME_NOT_OVER, TTT_TIE
from tqdm import trange

from experiments import get_experiment
from experiments.tic_tac_toe.minimax import evaluation_metrics, minimax_policy, value
from selfplay.checkpoints import load_checkpoint, save_checkpoint
from selfplay.evaluation import evaluate
from selfplay.ppo import make_batch, update
from selfplay.self_play import SelfPlay, random_policy
from selfplay.train import train


class ExperimentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        np.random.seed(7)
        torch.manual_seed(7)

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_batched_histories_and_updates_for_both_games(self):
        for name, shape, actions, max_moves in (
            ('connect4', (6, 7, 2), 7, 42), ('tic_tac_toe', (3, 3, 2), 9, 9),
        ):
            with self.subTest(experiment=name):
                experiment = get_experiment(name)
                agent = experiment.make_agent()
                pool = SelfPlay()
                pool.add_opponent(agent)
                frozen = {key: tensor.clone() for key, tensor in pool.opponents[0].state_dict().items()}
                envs = [experiment.make_env() for _ in range(8)]
                try:
                    records = []
                    with patch.object(pool, 'sample_opponent', return_value=(pool.opponents[0], 0)):
                        pairs = pool.play_games(envs, agent, 17, records=records)
                    self.assertEqual(len(pairs), 17)
                    self.assertEqual(len(records), 17)
                    self.assertEqual(pool.opponent_games[0].sum(), 17)
                    for pair in pairs:
                        self.assertEqual(sum(sum(t.reward) for t in pair), 0)
                        self.assertLessEqual(sum(len(t.action) for t in pair), max_moves)
                        for rollout in pair:
                            episode = rollout.get_episode()
                            n = len(rollout.action)
                            self.assertEqual(episode['obs']['observation'].shape, (n,) + shape)
                            self.assertEqual(episode['obs']['action_mask'].shape, (n, actions))
                            self.assertTrue(episode['terminated'][-1])
                            self.assertFalse(episode['terminated'][:-1].any())
                            self.assertFalse(episode['truncated'].any())
                            self.assertFalse(episode['reward'][:-1].any())
                            self.assertTrue(episode['obs']['action_mask'][np.arange(n), episode['action']].all())
                            np.testing.assert_array_equal(episode['next_obs']['observation'][:-1], episode['obs']['observation'][1:])
                            np.testing.assert_array_equal(episode['obs']['observation'][..., 0].sum(axis=(1, 2)), np.arange(n))
                        np.testing.assert_array_equal(pair[0].next_obs[-1]['observation'], pair[1].next_obs[-1]['observation'][..., ::-1])
                    for pair, record in zip(pairs, records):
                        self.assertEqual(len(record['actions']), sum(len(t.action) for t in pair))
                        replay = envs[0]
                        replay.reset()
                        current_name = replay.possible_agents[record['starting_idx']]
                        for action in record['actions']:
                            replay.step(action)
                        np.testing.assert_array_equal(replay.observe(current_name)['observation'], pair[0].next_obs[-1]['observation'])
                        self.assertEqual(replay.rewards[current_name], sum(pair[0].reward))
                    before = {key: tensor.clone() for key, tensor in agent.state_dict().items()}
                    args = experiment.default_args()
                    args.update_epochs = 1
                    metrics = update(agent, torch.optim.Adam(agent.parameters()), [current for current, _ in pairs], args)
                    self.assertTrue(all(np.isfinite(x) for x in metrics.values()))
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in agent.parameters()))
                    for prefix in ('encoder.', 'actor.', 'critic.'):
                        self.assertTrue(any(not torch.equal(before[k], v) for k, v in agent.state_dict().items() if k.startswith(prefix)))
                    self.assert_state_equal(frozen, pool.opponents[0].state_dict())
                finally:
                    for env in envs:
                        env.close()

    def test_opponent_sampling_uses_the_actual_starting_seat(self):
        pool = SelfPlay()
        pool.add_opponent(random_policy)
        pool.add_opponent(random_policy)
        pool.winrates = [np.array([0.9, 0.1]), np.array([0.1, 0.9])]
        for seat, expected in ((0, [0.1, 0.9]), (1, [0.9, 0.1])):
            with patch('selfplay.self_play.np.random.rand', return_value=0.5), patch('selfplay.self_play.np.random.choice', return_value=1) as choice:
                _, idx = pool.sample_opponent(seat)
            self.assertEqual(idx, 1)
            np.testing.assert_allclose(choice.call_args.kwargs['p'], expected)
        pool.winrates = [np.ones(2), np.ones(2)]
        with patch('selfplay.self_play.np.random.rand', return_value=0.5), patch('selfplay.self_play.np.random.choice', return_value=0) as choice:
            pool.sample_opponent(1)
        np.testing.assert_allclose(choice.call_args.kwargs['p'], [0.5, 0.5])

        experiment = get_experiment('tic_tac_toe')
        envs = [experiment.make_env() for _ in range(3)]
        try:
            for seat in (0, 1):
                previous = pool.opponent_games[1].copy()
                with patch('selfplay.self_play.np.random.randint', return_value=seat), patch.object(pool, 'sample_opponent', return_value=(pool.opponents[1], 1)) as sample:
                    pairs = pool.play_games(envs, random_policy, 7)
                self.assertTrue(all(call.args == (seat,) for call in sample.call_args_list))
                self.assertEqual(sample.call_count, 7)
                for current, _ in pairs:
                    self.assertEqual(int(current.obs[0]['observation'].sum()), seat)
                expected = previous.copy()
                expected[seat] += 7
                np.testing.assert_array_equal(pool.opponent_games[1], expected)
        finally:
            for env in envs:
                env.close()

    def test_pool_evicts_easiest_and_preserves_ids_and_statistics(self):
        agent = get_experiment('tic_tac_toe').make_agent()
        pool = SelfPlay(max_opponents=3)
        for _ in range(3):
            pool.add_opponent(agent)
        pool.winrates = [np.array([0.9, 0.1]), np.array([0.75, 0.85]), np.array([0.6, 0.6])]
        pool.opponent_games = [np.array([10, 20]), np.array([30, 40]), np.array([50, 60])]
        pool.add_opponent(agent)
        self.assertEqual(pool.opponent_ids, [0, 2, 3])
        np.testing.assert_allclose(pool.winrates, [[0.9, 0.1], [0.6, 0.6], [0.5, 0.5]])
        np.testing.assert_array_equal(pool.opponent_games, [[10, 20], [50, 60], [0, 0]])
        # Even when the new 0.5 prior is highest, retain the new snapshot.
        for wr, score in zip(pool.winrates, (0.1, 0.2, 0.3)):
            wr[:] = score
        pool.add_opponent(agent)
        self.assertEqual(pool.opponent_ids, [0, 2, 4])
        self.assertEqual(pool.next_opponent_id, 5)
        self.assertTrue(all(not p.requires_grad for opp in pool.opponents for p in opp.parameters()))

        env = get_experiment('tic_tac_toe').make_env()
        try:
            records = []
            with patch.object(pool, 'sample_opponent', return_value=(pool.opponents[2], 2)):
                pool.play_games([env], agent, 3, records=records)
            self.assertEqual([record['opponent'] for record in records], [4, 4, 4])
            self.assertEqual(pool.opponent_games[2].sum(), 3)
        finally:
            env.close()
        single = SelfPlay(max_opponents=1)
        single.add_opponent(agent)
        single.add_opponent(agent)
        self.assertEqual(single.opponent_ids, [1])

    def test_restore_smaller_pool_uses_saved_scores_and_legacy_ids(self):
        agent = get_experiment('tic_tac_toe').make_agent()
        pool = SelfPlay(max_opponents=5)
        for score in (0.1, 0.9, 0.4, 0.8, 0.2):
            pool.add_opponent(agent)
            pool.winrates[-1][:] = score
            pool.opponent_games[-1][:] = pool.opponent_ids[-1] + 10
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'pool.pt'
            save_checkpoint(path, agent, torch.optim.Adam(agent.parameters()), pool, 1, 0, 5)
            original = torch.load(path, weights_only=True)
            for legacy in (False, True):
                state = original.copy()
                if legacy:
                    del state['opponent_ids'], state['next_opponent_id']
                torch.save(state, path)
                restored = get_experiment('tic_tac_toe').make_agent()
                smaller = SelfPlay(max_opponents=2)
                load_checkpoint(path, restored, torch.optim.Adam(restored.parameters()), smaller)
                self.assertEqual(smaller.opponent_ids, [0, 4])
                self.assertEqual(smaller.next_opponent_id, 5)
                np.testing.assert_allclose(smaller.winrates, [[0.1, 0.1], [0.2, 0.2]])
                np.testing.assert_array_equal(smaller.opponent_games, [[10, 10], [14, 14]])
                smaller.add_opponent(restored)
                self.assertEqual(smaller.opponent_ids, [0, 5])

    def test_connect4_horizontal_symmetry(self):
        experiment = get_experiment('connect4')
        agent = experiment.make_agent()
        original, mirrored = experiment.make_env(), experiment.make_env()
        try:
            original.reset()
            mirrored.reset()
            # Fill opposite edge columns to test reflected illegal-action masks too.
            for _ in range(6):
                original.step(0)
                mirrored.step(6)
            obs, *_ = original.last()
            other, *_ = mirrored.last()
            np.testing.assert_array_equal(obs['observation'][:, ::-1], other['observation'])
            np.testing.assert_array_equal(obs['action_mask'][::-1], other['action_mask'])
            boards = torch.tensor(obs['observation'], dtype=torch.float32)[None]
            masks = torch.tensor(obs['action_mask'], dtype=torch.bool)[None]
            logits, values = agent.get_logits_and_value(boards)
            reflected_logits, reflected_values = agent.get_logits_and_value(boards.flip(2))
            torch.testing.assert_close(logits, reflected_logits.flip(-1))
            torch.testing.assert_close(values, reflected_values)
            torch.testing.assert_close(agent.get_logits(boards), logits)
            _, logprob, entropy, _ = agent.get_action_and_value(boards, masks, torch.tensor([1]))
            _, reflected_logprob, reflected_entropy, _ = agent.get_action_and_value(boards.flip(2), masks.flip(-1), torch.tensor([5]))
            torch.testing.assert_close(logprob, reflected_logprob)
            torch.testing.assert_close(entropy, reflected_entropy)
            empty = torch.zeros(3, 6, 7, 2)
            probabilities = agent.get_logits(empty).softmax(-1)
            torch.testing.assert_close(probabilities, probabilities.flip(-1))
            (logprob.sum() + values.sum()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in agent.parameters()))
        finally:
            original.close()
            mirrored.close()

    def test_tic_tac_toe_all_eight_symmetries_and_action_mapping(self):
        experiment = get_experiment('tic_tac_toe')
        agent = experiment.make_agent()
        self.assertEqual(len({tuple(p.tolist()) for p in agent.symmetry_indices}), 8)
        self.assertNotIn('symmetry_indices', agent.state_dict())
        sequences = ([], [0, 4, 1], [0, 4, 1, 2], [0, 4, 6, 3, 1])
        env = experiment.make_env()
        observations = []
        try:
            for moves in sequences:
                env.reset()
                for action in moves:
                    env.step(action)
                obs, *_ = env.last()
                observations.append(obs)
            boards = torch.tensor(np.stack([o['observation'] for o in observations]), dtype=torch.float32)
            masks = torch.tensor(np.stack([o['action_mask'] for o in observations]), dtype=torch.bool)
            logits, values = agent.get_logits_and_value(boards)
            torch.testing.assert_close(logits, agent.get_logits(boards))
            probabilities = logits.masked_fill(~masks, -torch.inf).softmax(-1)
            for permutation, inverse in zip(agent.symmetry_indices, agent.inverse_indices):
                torch.testing.assert_close(permutation[inverse], torch.arange(9))
                transformed = boards.reshape(-1, 9, 2)[:, permutation].reshape(-1, 3, 3, 2)
                transformed_masks = masks[:, permutation]
                other_logits, other_values = agent.get_logits_and_value(transformed)
                torch.testing.assert_close(other_logits, logits[:, permutation])
                torch.testing.assert_close(other_values, values)
                other_probabilities = other_logits.masked_fill(~transformed_masks, -torch.inf).softmax(-1)
                torch.testing.assert_close(other_probabilities, probabilities[:, permutation])
                # Replay transformed actions in PettingZoo itself, not just array math.
                for i, moves in enumerate(sequences):
                    env.reset()
                    for action in moves:
                        env.step(int(inverse[action]))
                    obs, *_ = env.last()
                    np.testing.assert_array_equal(obs['observation'], transformed[i].numpy())
                    np.testing.assert_array_equal(obs['action_mask'], transformed_masks[i].numpy())
                actions = torch.tensor([int(np.flatnonzero(o['action_mask'])[0]) for o in observations])
                _, logprob, entropy, _ = agent.get_action_and_value(boards, masks, actions)
                _, other_logprob, other_entropy, _ = agent.get_action_and_value(transformed, transformed_masks, inverse[actions])
                torch.testing.assert_close(logprob, other_logprob)
                torch.testing.assert_close(entropy, other_entropy)
            empty_probabilities = probabilities[0]
            torch.testing.assert_close(empty_probabilities[[0, 2, 6, 8]], empty_probabilities[0].expand(4))
            torch.testing.assert_close(empty_probabilities[[1, 3, 5, 7]], empty_probabilities[1].expand(4))
            (logprob.sum() + values.sum()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in agent.parameters()))
            sampled = agent.act_batch(observations)
            self.assertTrue(all(obs['action_mask'][a] for obs, a in zip(observations, sampled)))
        finally:
            env.close()

    def test_changed_policy_version_resets_old_sampling_statistics(self):
        experiment = get_experiment('tic_tac_toe')
        agent = experiment.make_agent()
        pool = SelfPlay()
        pool.add_opponent(agent)
        pool.winrates[0][:] = [0.9, 0.1]
        pool.opponent_games[0][:] = [100, 200]
        pool.random_winrates[:] = [0.95, 0.8]
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'before_symmetry.pt'
            save_checkpoint(path, agent, torch.optim.Adam(agent.parameters()), pool, 10, 5, 300)
            state = torch.load(path, weights_only=True)
            del state['policy_version']
            torch.save(state, path)
            restored = experiment.make_agent()
            restored_pool = SelfPlay()
            with self.assertWarnsRegex(UserWarning, 'Policy behavior changed'):
                load_checkpoint(path, restored, torch.optim.Adam(restored.parameters()), restored_pool, experiment.make_agent)
            self.assert_state_equal(agent.state_dict(), restored.state_dict())
            np.testing.assert_array_equal(restored_pool.winrates, [[0.5, 0.5]])
            np.testing.assert_array_equal(restored_pool.opponent_games, [[0, 0]])
            np.testing.assert_array_equal(restored_pool.random_winrates, [0.5, 0.5])

    def test_checkpoint_resume_and_lowest_winrate_logging(self):
        for name in ('connect4', 'tic_tac_toe'):
            with self.subTest(experiment=name), TemporaryDirectory() as temp:
                args = get_experiment(name).default_args()
                args.seed, args.num_iterations, args.games_per_update = 3, 4, 5
                args.num_envs, args.eval_every, args.update_epochs, args.save_every = 3, 0, 1, 2
                args.max_opponents = 2  # Exercise eviction during training and resume.
                args.checkpoint_dir = str(Path(temp) / 'full')
                run = MagicMock()
                run.id = 'test'
                context = MagicMock()
                context.__enter__.return_value = run
                with patch('selfplay.train.wandb.init', return_value=context) as init, patch('selfplay.train.trange', side_effect=lambda *a, **k: trange(*a, **k, disable=True)):
                    train(args)
                    self.assertEqual(init.call_args.kwargs['group'], name)
                    full_path = next(Path(args.checkpoint_dir).glob('*/iteration_4.pt'))
                    self.assertNotIn('name', init.call_args.kwargs)
                    self.assertNotIn('run_datetime', init.call_args.kwargs['config'])
                    self.assertRegex(full_path.parent.name, r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}$')
                    full = torch.load(full_path, weights_only=True)
                    for call in run.log.call_args_list:
                        metrics = call.args[0]
                        self.assertTrue(0 <= metrics['train/second_player_fraction'] <= 1)
                        rates = {int(k.split('_')[-1]): v for k, v in metrics.items() if k.startswith('winrate/opponent_')}
                        lowest = min(rates, key=rates.get)
                        self.assertEqual(metrics['train/lowest_winrate_opponent'], lowest)
                        self.assertEqual(metrics['train/lowest_winrate'], rates[lowest])
                    args.checkpoint = str(full_path.with_name('iteration_2.pt'))
                    args.checkpoint_dir = str(Path(temp) / 'resumed')
                    train(args)
                    resumed = torch.load(next(Path(args.checkpoint_dir).glob('*/iteration_4.pt')), weights_only=True)
                    self.assert_state_equal(full, resumed)

    def test_old_connect4_checkpoint_and_cross_game_rejection(self):
        connect = get_experiment('connect4').make_agent()
        pool = SelfPlay()
        pool.add_opponent(connect)
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'old_connect4.pt'
            save_checkpoint(path, connect, torch.optim.Adam(connect.parameters()), pool, 1, 0, 5)
            state = torch.load(path, weights_only=True)
            del state['policy_class']  # Previous Connect 4 checkpoint format.
            torch.save(state, path)
            restored = get_experiment('connect4').make_agent()
            load_checkpoint(path, restored, torch.optim.Adam(restored.parameters()), SelfPlay())
            self.assert_state_equal(connect.state_dict(), restored.state_dict())
            tic = get_experiment('tic_tac_toe').make_agent()
            with self.assertRaisesRegex(ValueError, 'architecture|policy'):
                load_checkpoint(path, tic, torch.optim.Adam(tic.parameters()), SelfPlay())

    def test_bring_your_own_policy_and_factory_resume(self):
        # No Policy base class, CNN, encoder, actor, critic, or shape attributes.
        class StandalonePolicy(nn.Module):
            def __init__(self, width):
                super().__init__()
                self.network = nn.Sequential(nn.Flatten(), nn.Linear(18, width), nn.Tanh())
                self.move_scores = nn.Linear(width, 9)
                self.state_score = nn.Linear(width, 1)

            def get_action_and_value(self, obs, masks, action=None):
                features = self.network(obs)
                distribution = Categorical(logits=self.move_scores(features).masked_fill(~masks, -torch.inf))
                action = distribution.sample() if action is None else action
                return action, distribution.log_prob(action), distribution.entropy(), self.state_score(features).squeeze(-1)

            @torch.inference_mode()
            def act_batch(self, observations):
                obs = torch.tensor(np.stack([o['observation'] for o in observations]), dtype=torch.float32)
                masks = torch.tensor(np.stack([o['action_mask'] for o in observations]), dtype=torch.bool)
                logits = self.move_scores(self.network(obs)).masked_fill(~masks, -torch.inf)
                return torch.multinomial(logits.softmax(-1), 1).squeeze(-1).numpy()

            def forward(self, obs):
                return int(self.act_batch([obs])[0])

        experiment = get_experiment('tic_tac_toe')
        factory = MagicMock(side_effect=lambda: StandalonePolicy(width=23))
        custom = SimpleNamespace(make_agent=factory, make_env=experiment.make_env)
        with TemporaryDirectory() as temp:
            args = experiment.default_args()
            args.seed, args.num_iterations, args.games_per_update = 3, 4, 7
            args.num_envs, args.eval_every, args.update_epochs, args.save_every = 3, 0, 1, 2
            args.checkpoint_dir = str(Path(temp) / 'full')
            run, context = MagicMock(), MagicMock()
            run.id = 'test'
            context.__enter__.return_value = run
            with patch('selfplay.train.get_experiment', return_value=custom), patch('selfplay.train.wandb.init', return_value=context), patch('selfplay.train.trange', side_effect=lambda *a, **k: trange(*a, **k, disable=True)):
                trained = train(args)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in trained.parameters()))
                full_path = next(Path(args.checkpoint_dir).glob('*/iteration_4.pt'))
                expected = torch.load(full_path, weights_only=True)
                args.checkpoint = str(full_path.with_name('iteration_2.pt'))
                args.checkpoint_dir = str(Path(temp) / 'resumed')
                train(args)
                resumed = torch.load(next(Path(args.checkpoint_dir).glob('*/iteration_4.pt')), weights_only=True)
                self.assert_state_equal(expected, resumed)
                self.assertGreaterEqual(factory.call_count, 5)

    def test_minimax_matches_pettingzoo_on_every_reachable_position(self):
        positions = {}

        @cache
        def reference(squares, player):
            board = Board()
            board.squares = list(squares)
            status = board.game_status()
            perspective = tuple(0 if mark == 0 else 1 if mark == player + 1 else -1 for mark in squares)
            if status != TTT_GAME_NOT_OVER:
                expected = 0 if status == TTT_TIE else 1 if status == player else -1
            else:
                scores = []
                for action in board.legal_moves():
                    child = list(squares)
                    child[action] = player + 1
                    scores.append(-reference(tuple(child), 1 - player))
                expected = max(scores)
                positions[(squares, player)] = expected
            self.assertEqual(value(perspective), expected)
            return expected

        self.assertEqual(reference((0,) * 9, 0), 0)
        self.assertEqual(reference.cache_info().currsize, 5478)
        for (squares, player), expected in positions.items():
            marks = np.array(squares).reshape(3, 3)
            obs = {'observation': np.stack([marks == player + 1, marks == 2 - player], axis=-1).astype(np.int8),
                   'action_mask': np.array(squares) == 0}
            action = minimax_policy(obs)
            child = list(squares)
            child[action] = player + 1
            self.assertEqual(-reference(tuple(child), 1 - player), expected)
        metrics = evaluation_metrics(minimax_policy, games_per_seat=10)
        self.assertEqual(metrics['eval/minimax_draw_rate_first'], 1)
        self.assertEqual(metrics['eval/minimax_draw_rate_second'], 1)
        self.assertEqual(metrics['eval/minimax_loss_rate'], 0)

    def test_video_evaluation_preserves_training_state(self):
        for name in ('connect4', 'tic_tac_toe'):
            with self.subTest(experiment=name):
                experiment = get_experiment(name)
                agent = experiment.make_agent()
                pool = SelfPlay()
                pool.add_opponent(agent)
                pool.add_opponent(agent)
                pool.winrates = [np.array([0.2, 0.8]), np.array([0.7, 0.3])]
                pool.opponent_games = [np.array([13, 9]), np.array([5, 21])]
                numpy_state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
                original_sampler = SelfPlay.sample_opponent

                def sample(collector, seat):
                    self.assertIsNot(collector, pool)
                    np.testing.assert_array_equal(collector.winrates, pool.winrates)
                    return original_sampler(collector, seat)

                with patch.object(SelfPlay, 'sample_opponent', autospec=True, side_effect=sample) as sampler:
                    metrics = evaluate(agent, pool, experiment, num_games=9, num_envs=4)
                self.assertEqual(sampler.call_count, 9)
                self.assertEqual(set(metrics), {'eval/mean_reward', 'eval/video', 'eval/win_rate_first', 'eval/win_rate_second', 'eval/second_player_fraction'})
                self.assertTrue(0 <= metrics['eval/second_player_fraction'] <= 1)
                self.assertTrue(0 <= metrics['eval/win_rate_first'] <= 1)
                self.assertTrue(0 <= metrics['eval/win_rate_second'] <= 1)
                self.assertTrue(Path(metrics['eval/video']._path).is_file())
                self.assertTrue(agent.training)
                self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
                actual = np.random.get_state()
                np.testing.assert_array_equal(numpy_state[1], actual[1])
                self.assertEqual(numpy_state[2:], actual[2:])
                np.testing.assert_array_equal(pool.winrates, [[0.2, 0.8], [0.7, 0.3]])
                np.testing.assert_array_equal(pool.opponent_games, [[13, 9], [5, 21]])

    def test_evaluation_with_no_frozen_opponents(self):
        tic = get_experiment('tic_tac_toe')
        experiment = SimpleNamespace(make_env=tic.make_env, player_labels=tic.player_labels)
        metrics = evaluate(tic.make_agent(), SelfPlay(), experiment, num_games=5, num_envs=3)
        self.assertTrue(Path(metrics['eval/video']._path).is_file())

    def test_evaluation_metrics_by_seat_and_sampled_video(self):
        tic = get_experiment('tic_tac_toe')
        pairs = [(SimpleNamespace(reward=[r]), None) for r in (-1, 1, 0, 1)]
        records = [{'opponent': 0, 'starting_idx': seat, 'actions': []} for seat in (0, 0, 1, 1)]

        def play(collector, envs, agent, num_games, *, records):
            records.extend([{'opponent': 0, 'starting_idx': seat, 'actions': []} for seat in (0, 0, 1, 1)])
            return pairs

        with patch.object(SelfPlay, 'play_games', autospec=True, side_effect=play), patch('selfplay.evaluation.game_video', return_value='video') as video, patch('selfplay.evaluation.np.random.choice', return_value=2) as choice:
            metrics = evaluate(tic.make_agent(), SelfPlay(), tic, num_games=4, num_envs=2)
        self.assertEqual(metrics, {'eval/mean_reward': 0.25, 'eval/video': 'video', 'eval/win_rate_first': 0.5, 'eval/win_rate_second': 0.5, 'eval/second_player_fraction': 0.5})
        np.testing.assert_array_equal(choice.call_args.args[0], np.arange(4))
        self.assertEqual(video.call_args.args[1], records[2])
        self.assertEqual(video.call_args.args[2], 0)

    def test_second_player_win_loss_draw_and_ppo_reward_signal(self):
        experiment = get_experiment('tic_tac_toe')
        agent = experiment.make_agent()
        # Zero baseline isolates the sign of the terminal reward in GAE.
        with torch.no_grad():
            for parameter in agent.critic.parameters():
                parameter.zero_()
        env = experiment.make_env()

        def scripted(actions):
            moves = iter(actions)
            return lambda obs: next(moves)

        try:
            for ours, theirs, outcome in (
                ([4, 1], [6, 0, 3], -1),
                ([3, 4, 5], [0, 1, 8], 1),
                ([1, 4, 5, 6], [0, 2, 3, 7, 8], 0),
            ):
                current, opponent = SelfPlay().play_a_game(
                    env, scripted(ours), scripted(theirs), starting_idx=1,
                )
                self.assertEqual(current.action, ours)
                self.assertEqual(int(current.obs[0]['observation'].sum()), 1)
                self.assertEqual(current.reward[-1], outcome)
                self.assertEqual(sum(opponent.reward), -outcome)
                self.assertTrue(current.terminated[-1])
                batch = make_batch(agent, [current], experiment.default_args())
                self.assertAlmostEqual(batch[4][-1].item(), outcome)
                self.assertAlmostEqual(batch[5][-1].item(), outcome)
                if outcome:
                    obs, masks, actions, _, advantages, _ = batch
                    logits = agent.get_logits(obs[-1:]).masked_fill(~masks[-1:], -torch.inf)
                    loss = -Categorical(logits=logits).log_prob(actions[-1:]) * advantages[-1:]
                    gradient = torch.autograd.grad(loss.sum(), logits)[0]
                    # Gradient descent discourages the losing move, encourages a winning one.
                    self.assertLess(gradient[0, actions[-1]].item() * outcome, 0)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
