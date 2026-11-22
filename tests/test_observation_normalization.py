"""Observation normalization: statistics, PPO consistency, checkpoints and worker snapshots."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from checkpoints import load_policy
from teacher_model import RunningObservationStatistics, TeacherActorCritic
from teacher_ppo import TeacherPPOAgent, TeacherRolloutBuffer
from teacher_rollouts import _copy_parameters, snapshot_tensors


def filled_rollout(steps=64, seed=0):
    rng = np.random.default_rng(seed)
    rollout = TeacherRolloutBuffer(steps, 2, 48, 2)
    for _ in range(steps):
        rollout.add(rng.normal(3.0, 10.0, (2, 48)).astype(np.float32),
                    rng.normal(size=(2, 2)).astype(np.float32),
                    rng.choice([0.0, 0.1], size=2).astype(np.float32),
                    np.zeros(2, np.float32), np.full(2, -1.0, np.float32), np.zeros(2, np.float32),
                    np.zeros(2, bool), np.zeros(2, bool))
    return rollout


class RunningStatisticsTests(unittest.TestCase):
    def test_merged_batches_match_full_data(self):
        rng = np.random.default_rng(1)
        data = rng.normal(5.0, 7.0, size=(300, 48))
        stats = RunningObservationStatistics()
        for chunk in np.array_split(data, 7):
            stats.update(chunk)
        blocks = data.reshape(-1, 24)
        np.testing.assert_allclose(stats.mean, blocks.mean(0), rtol=1e-10)
        np.testing.assert_allclose(stats.var, blocks.var(0), rtol=1e-10)
        self.assertEqual(stats.count, 600)

    def test_both_halves_share_statistics(self):
        stats = RunningObservationStatistics()
        stats.update(np.random.default_rng(2).normal(size=(50, 48)))
        mean, var = stats.model_statistics()
        np.testing.assert_array_equal(mean[:24], mean[24:])
        np.testing.assert_array_equal(var[:24], var[24:])


class ModelTests(unittest.TestCase):
    def test_disabled_model_is_unchanged(self):
        model = TeacherActorCritic(48, 2, hidden_size=8)
        self.assertEqual(len(list(model.buffers())), 0)
        states = torch.randn(5, 48) * 20
        self.assertTrue(torch.equal(model.normalize(states), states))

    def test_normalization_standardizes_and_clips(self):
        model = TeacherActorCritic(48, 2, hidden_size=8, normalize_observations=True)
        model.set_observation_statistics(np.full(48, 2.0), np.full(48, 4.0))
        out = model.normalize(torch.tensor([[4.0] * 48, [100.0] * 48]))
        torch.testing.assert_close(out[0], torch.ones(48), atol=1e-6, rtol=0)
        torch.testing.assert_close(out[1], torch.full((48,), 5.0))


class MeanActionTests(unittest.TestCase):
    def test_mean_action_matches_forward_and_uses_normalization(self):
        model = TeacherActorCritic(48, 2, hidden_size=8, normalize_observations=True)
        model.set_observation_statistics(np.full(48, 3.0), np.full(48, 16.0))
        states = torch.randn(6, 48) * 10
        means, _ = model(states)
        torch.testing.assert_close(model.mean_action(states), torch.tanh(means))
        raw = torch.tanh(model.actor(states))
        self.assertFalse(torch.allclose(model.mean_action(states), raw))


class PPOConsistencyTests(unittest.TestCase):
    def test_statistics_change_only_after_the_update(self):
        torch.manual_seed(0)
        agent = TeacherPPOAgent(48, 2, hidden_size=8, normalize_observations=True,
                                update_epochs=1, minibatch_size=32)
        rollout = filled_rollout()
        before = agent.model.observation_mean.clone()
        seen = []
        original = agent.model.normalize

        def spy(states):
            seen.append(agent.model.observation_mean.clone())
            return original(states)
        agent.model.normalize = spy
        agent.update(rollout)
        # Every forward pass inside the update used the collection-time statistics.
        self.assertTrue(all(torch.equal(m, before) for m in seen))
        expected = rollout.states.reshape(-1, 24).mean(0)
        np.testing.assert_allclose(agent.model.observation_mean.numpy()[:24], expected, rtol=1e-5)


class PersistenceTests(unittest.TestCase):
    def test_checkpoint_round_trip_keeps_statistics(self):
        model = TeacherActorCritic(48, 2, hidden_size=8, normalize_observations=True)
        model.set_observation_statistics(np.arange(48.0), np.full(48, 9.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            torch.save(dict(model_state_dict=model.state_dict(), state_size=48, action_size=2,
                            hidden_size=8, observation_order="self_then_other", config={},
                            normalize_observations=True), path)
            loaded, _, kind = load_policy(path)
        self.assertEqual(kind, "teacher")
        self.assertTrue(torch.equal(loaded.observation_mean, model.observation_mean))
        states = torch.randn(3, 48) * 10
        torch.testing.assert_close(loaded(states)[0], model(states)[0])

    def test_old_checkpoints_load_without_normalization(self):
        model = TeacherActorCritic(48, 2, hidden_size=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            torch.save(dict(model_state_dict=model.state_dict(), state_size=48, action_size=2,
                            hidden_size=8, observation_order="self_then_other", config={}), path)
            loaded, _, _ = load_policy(path)
        self.assertFalse(loaded.normalize_observations)

    def test_worker_snapshot_includes_statistics(self):
        parent = TeacherActorCritic(48, 2, hidden_size=8, normalize_observations=True)
        parent.set_observation_statistics(np.full(48, 1.5), np.full(48, 2.5))
        flat = np.concatenate([t.detach().numpy().reshape(-1) for t in snapshot_tensors(parent)])
        shared = flat.astype(np.float32).tobytes()
        worker = TeacherActorCritic(48, 2, hidden_size=8, normalize_observations=True)
        _copy_parameters(worker, bytearray(shared))
        states = torch.randn(4, 48)
        torch.testing.assert_close(worker(states)[0], parent(states)[0])
        self.assertTrue(torch.equal(worker.observation_var, parent.observation_var))


if __name__ == "__main__":
    unittest.main()
