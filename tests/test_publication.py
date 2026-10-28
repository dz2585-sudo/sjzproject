import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from aimrl.model_io import artifact_digest, load_model_array, resolve_artifact
from aimrl.provenance import verify_historical_hashes


class PublicationTests(unittest.TestCase):
    def setUp(self):
        Path('.tmp').mkdir(exist_ok=True)

    def test_lossless_model_package_and_missing_npy_resolution(self):
        with tempfile.TemporaryDirectory(dir=Path('.tmp').resolve()) as directory:
            root = Path(directory)
            data = np.arange(120, dtype=np.float32).reshape(12, 10)
            stream = io.BytesIO(); np.save(stream, data, allow_pickle=False)
            raw = stream.getvalue()
            with zipfile.ZipFile(root / 'q_table.npz', 'w', zipfile.ZIP_DEFLATED) as package:
                package.writestr('q_table.npy', raw)
            self.assertEqual(resolve_artifact('q_table.npy', root), root / 'q_table.npz')
            self.assertEqual(artifact_digest('q_table.npy', root), hashlib.sha256(raw).hexdigest())
            np.testing.assert_array_equal(load_model_array('q_table.npz', root), data)

    def test_historical_path_maps_to_published_artifact(self):
        with tempfile.TemporaryDirectory(dir=Path('.tmp').resolve()) as directory:
            root = Path(directory); (root / 'provenance').mkdir(); (root / 'results').mkdir()
            (root / 'results/record.csv').write_bytes(b'unchanged numeric record\n')
            manifest = {'source_root': 'C:/historical/project', 'artifacts': {'outputs/record.csv': {'published_path': 'results/record.csv'}}}
            (root / 'provenance/archive_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            self.assertEqual(resolve_artifact('C:/historical/project/outputs/record.csv', root), root / 'results/record.csv')

    def test_missing_historical_evidence_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory(dir=Path('.tmp').resolve()) as directory:
            with self.assertRaisesRegex(ValueError, 'Unverified'):
                verify_historical_hashes(Path(directory), {'missing.json': '0' * 64})
