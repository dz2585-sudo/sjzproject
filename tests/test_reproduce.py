"""Integration checks for the supported checkout reproduction entry point."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aimrl import reproduce as new
from aimrl.tracking_research import workflow as archived
from aimrl.tracking_research.io import ROOT, read_json


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(ROOT / 'configs/tracking_research.json')
        temporary_root = ROOT / '.tmp'
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_output_protection_and_partial_preservation(self):
        for p in (ROOT, ROOT / 'results/new', ROOT / 'aimrl/new', ROOT / 'provenance/new'):
            with self.assertRaises(ValueError):
                new.fresh_output(p, ROOT / 'results/tracking_improvement')
        out = new.fresh_output(self.root / 'session', ROOT / 'results/tracking_improvement')
        (out / 'partial.txt').write_text('keep')
        with self.assertRaises(FileExistsError):
            new.fresh_output(out, ROOT / 'results/tracking_improvement')
        self.assertEqual((out / 'partial.txt').read_text(), 'keep')

    def test_telemetry_failure_is_nonfatal(self):
        with patch.object(new.sys, 'platform', 'win32'), patch(
                'aimrl.tracking_research.io.peak_memory_bytes', side_effect=OSError):
            self.assertEqual(new.memory_record(), {'available': False, 'bytes': None, 'reason': 'OSError'})
        record = new.memory_record()
        self.assertIn('available', record)
        if record['available']:
            self.assertGreater(record['bytes'], 0)

    def test_short_runs_match_archived_learning_for_all_four_methods(self):
        c = copy.deepcopy(self.config)
        c['study']['pilot_steps'] = 1001
        for method in new.METHODS:
            reference = self.root / 'reference' / method
            with patch.object(archived, 'peak_memory_bytes', return_value=0):
                archived.train_run(c, method, 99, reference, self.root, pilot=True)
            new.train_one(self.config, method, 99, self.root / 'new', smoke=True)
            trial = self.root / 'new/runs' / method / 'seed_99'
            for name, member in [('q_table.npz', 'q_table'), ('visitation.npz', 'counts')]:
                with np.load(reference / name) as left, np.load(trial / name) as right:
                    np.testing.assert_array_equal(left[member], right[member])
            self.assertEqual((reference / 'training_windows.csv').read_bytes(),
                             (trial / 'training_windows.csv').read_bytes())
            meta = read_json(trial / 'metadata.json')
            self.assertEqual((meta['complete_windows'], meta['partial_ticks']), (3, 101))

    def test_load_replay_isolation_and_checksum_rejection(self):
        new.train_one(self.config, 'relative_potential', 99, self.root, smoke=True)
        a, _ = new.load_checked_model(self.config, 'relative_potential', 99, self.root)
        before = a.snapshot()
        summary, _, _ = new.evaluate(self.config, 'relative_potential', [3700000], a)
        self.assertEqual(before, a.snapshot())
        clarified = new.clarified_summary(summary)
        self.assertNotIn('fully_blocked_move_fraction', summary['slices']['all'])
        self.assertEqual(clarified['slices']['all']['fully_blocked_move_fraction'],
                         summary['slices']['all']['clipped_move_fraction'])
        path = self.root / 'runs/relative_potential/seed_99/q_table.npz'
        path.write_bytes(path.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            new.load_checked_model(self.config, 'relative_potential', 99, self.root)

    def test_cli_train_then_evaluate_from_external_directory(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        train = self.root / 'cli_train'
        evaluation = self.root / 'cli_evaluation'
        commands = [
            ['--mode', 'train', '--smoke', '--seeds', '99', '--output', str(train)],
            ['--mode', 'evaluate', '--model-root', str(train), '--seeds', '99',
             '--split', 'development', '--limit', '2', '--output', str(evaluation)]]
        for args in commands:
            p = subprocess.run([sys.executable, '-m', 'aimrl.reproduce', *args],
                               cwd=self.root, env=env, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        record = read_json(evaluation / 'session.json')
        self.assertTrue(record['complete'])
        self.assertEqual(record['evaluation']['evaluation_interactions'], 1800)
        self.assertFalse(record['evaluation']['recorded_comparison_performed'])
        self.assertTrue((evaluation / 'comparison.png').is_file())
        stats = read_json(evaluation / 'cross_seed_summary.json')
        self.assertIsNone(stats['relative_potential']['mean_distance']['sample_sd'])

    def test_failed_session_is_recorded(self):
        out = self.root / 'failed'
        with self.assertRaises(FileNotFoundError):
            new.main(['--mode', 'evaluate', '--model-root', str(self.root / 'missing'),
                      '--limit', '1', '--output', str(out)])
        record = read_json(out / 'session.json')
        self.assertFalse(record['complete'])
        self.assertTrue(record['error'].startswith('FileNotFoundError:'))

    def test_unknown_source_change_is_rejected(self):
        with patch.object(new, 'source_hash_matches', return_value=False):
            with self.assertRaisesRegex(ValueError, 'Unverified published source'):
                new.checked_config()


if __name__ == '__main__':
    unittest.main()
