"""Exercise stopping and milestone policies without launching Unity."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import train_teacher as trainer


class ScorePool:
    def __init__(self, *args):
        self.calls = 0

    def collect(self, steps):
        self.calls += 1
        score = 0.6 if self.calls == 1 else 1.2
        episodes = [dict(rollout_step=i + 1, steps=1, return_agent0=score,
                         return_agent1=0., score=score, terminated=True,
                         environment_index=0, environment_episode=(self.calls-1)*100+i+1,
                         time_limit=False) for i in range(100)]
        return SimpleNamespace(position=100), episodes

    def close(self):
        pass


class TeacherTargetTests(unittest.TestCase):
    def config(self, root, target=None):
        c = dict(name='test', status='prepared', seed=2, worker_id=1, iterations=3,
                 rollout_steps=100, hidden_size=4, environment='tennis', checkpoint_every=1,
                 hyperparameters=dict(learning_rate=.0003, gamma=.995, clip_epsilon=.2,
                                      update_epochs=4, minibatch_size=512, value_loss_coef=.5,
                                      gae_lambda=.95))
        if target is not None:
            c['target_score'] = target
        p = root/'runs/test/config.json'
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(c))
        (root/'tennis').touch()
        return p, c

    def run_case(self, target=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p, _ = self.config(root, target)
            def update(agent, rollout):
                with torch.no_grad():
                    agent.model.log_std.add_(.1)
                return dict(policy_loss=0., value_loss=0., optimizer_steps=1)
            with patch.object(trainer, 'ROOT', root), patch.object(trainer, 'TeacherRolloutPool', ScorePool), \
                    patch.object(trainer.TeacherPPOAgent, 'update', update):
                trainer.train_from_config(p)
            config = json.loads(p.read_text())
            final = torch.load(p.parent/'checkpoint.pt', weights_only=True)
            milestones = {}
            for item in config.get('score_milestones', []):
                milestones[item['threshold']] = (item, torch.load(root/item['checkpoint'], weights_only=True))
            return config, final, milestones

    def test_target_one_keeps_learning_after_solve_and_saves_both_policies(self):
        c, final, milestones = self.run_case(1.)
        self.assertEqual(c['solved_episode'], 100)
        self.assertEqual(c['solved_environment_steps'], 100)
        self.assertEqual(c['iterations_completed'], 1)
        self.assertEqual(c['target_episode'], 167)
        self.assertEqual(c['target_environment_steps'], 167)
        self.assertEqual(c['stop_reason'], 'target_score')
        self.assertEqual(set(milestones), {.5, 1.})
        self.assertEqual(milestones[.5][0]['episode'], 100)
        self.assertEqual(milestones[1.][0]['episode'], 167)
        self.assertFalse(torch.equal(milestones[.5][1]['model_state_dict']['log_std'], final['model_state_dict']['log_std']))
        torch.testing.assert_close(milestones[1.][1]['model_state_dict']['log_std'], final['model_state_dict']['log_std'])

    def test_default_still_stops_at_half_without_updating(self):
        c, _, milestones = self.run_case()
        self.assertEqual(c['solved_episode'], 100)
        self.assertEqual(c['iterations_completed'], 0)
        self.assertEqual(c['stop_reason'], 'solved')
        self.assertEqual(set(milestones), {.5})

    def test_rejects_invalid_targets_before_startup(self):
        for value in (0, -.1, True, float('nan'), float('inf'), '1'):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                p, c = self.config(root, value)
                with patch.object(trainer, 'ROOT', root), self.assertRaises(ValueError):
                    trainer.validate_config(p, c)

    def test_nearby_thresholds_have_distinct_checkpoint_paths(self):
        _, _, milestones = self.run_case(1.000001)
        paths = [item['checkpoint'] for item, _ in milestones.values()]
        self.assertEqual(len(paths), len(set(paths)))


if __name__ == '__main__':
    unittest.main()
