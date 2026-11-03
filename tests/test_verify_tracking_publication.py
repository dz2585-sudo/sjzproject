"""Exercise publication verification and checkout failure records in both Python modes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path('provenance/tracking_improvement_publication.json')
REQUIRED_FILE = Path('results/tracking_improvement/evaluation/final/rule_windows.csv')


class PublicationVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary_root = ROOT / '.tmp'
        temporary_root.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        cls.addClassCleanup(temporary.cleanup)
        cls.checkout = Path(temporary.name)
        for directory in ('aimrl', 'configs', 'docs', 'provenance', 'results', 'scripts', 'tests'):
            shutil.copytree(ROOT / directory, cls.checkout / directory,
                            ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('README.md', 'spec.md', 'requirements.txt', 'requirements-reproduction.txt'):
            shutil.copyfile(ROOT / name, cls.checkout / name)

    def verify_case(self, name, expected_error=None):
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
        env.pop('PYTHONOPTIMIZE', None)
        for flags in ((), ('-O',)):
            mode = 'optimized' if flags else 'normal'
            with self.subTest(mode=mode):
                output = Path('.tmp') / f'{name}-{mode}'
                verification = self.checkout / output / 'direct.json'
                result = subprocess.run(
                    [sys.executable, *flags, 'scripts/verify_tracking_publication.py',
                     '--output', str(output / 'direct.json')],
                    cwd=self.checkout, env=env, capture_output=True, text=True,
                    encoding='utf-8', timeout=180,
                )
                diagnostic = result.stdout + result.stderr
                if expected_error is None:
                    self.assertEqual(result.returncode, 0, diagnostic)
                    evidence = json.loads(verification.read_text(encoding='utf-8'))
                    self.assertIs(evidence['complete'], True)
                    self.assertEqual(evidence['formal_runs'], 12)
                    self.assertEqual(len(evidence['gifs']), 4)
                    self.assertEqual(json.loads(result.stdout), evidence)
                else:
                    self.assertEqual(result.returncode, 1, diagnostic)
                    self.assertIn(expected_error, result.stderr)
                    self.assertFalse(verification.exists(), diagnostic)
                    self.assertNotIn('"complete": true', result.stdout)
                    self.check_failed_session(flags, output / 'session', env, expected_error)

    def check_failed_session(self, flags, output, env, expected_error):
        result = subprocess.run(
            [sys.executable, *flags, '-m', 'aimrl.reproduce', '--mode', 'verify', '--output', str(output)],
            cwd=self.checkout, env=env, capture_output=True, text=True, encoding='utf-8', timeout=180,
        )
        diagnostic = result.stdout + result.stderr
        self.assertEqual(result.returncode, 1, diagnostic)
        self.assertIn(expected_error, result.stderr)
        evidence = json.loads((self.checkout / output / 'session.json').read_text(encoding='utf-8'))
        self.assertIs(evidence['complete'], False)
        self.assertIn('error', evidence)
        self.assertFalse((self.checkout / output / 'publication_verification.json').exists(), diagnostic)
        self.assertNotIn('"complete": true', result.stdout)

    def test_valid_publication(self):
        self.verify_case('valid')

    def test_wrong_protocol_digest(self):
        path = self.checkout / MANIFEST
        original = path.read_bytes()
        data = json.loads(original)
        data['study_protocol_sha256'] = '0' * 64
        try:
            path.write_text(json.dumps(data), encoding='utf-8')
            self.verify_case('wrong-protocol-digest', 'SHA-256 mismatch: results')
        finally:
            path.write_bytes(original)

    def test_missing_required_file(self):
        path = self.checkout / REQUIRED_FILE
        backup = path.with_suffix('.backup')
        path.rename(backup)
        try:
            self.verify_case('missing-file', 'rule_windows.csv')
        finally:
            backup.rename(path)

    def test_modified_required_file(self):
        path = self.checkout / REQUIRED_FILE
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'\n')
            self.verify_case('modified-file', 'SHA-256 mismatch:')
        finally:
            path.write_bytes(original)
