import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from aimrl.provenance import source_hash_matches


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        Path('.tmp').mkdir(exist_ok=True)

    def test_original_source_requires_no_translation_record(self):
        with tempfile.TemporaryDirectory(dir=Path('.tmp').resolve()) as directory:
            root = Path(directory)
            (root / 'example.py').write_bytes(b'original source')
            recorded = hashlib.sha256(b'original source').hexdigest()
            self.assertTrue(source_hash_matches(root, 'example.py', recorded))
            self.assertFalse(source_hash_matches(root, 'missing.py', recorded))

    def test_only_exact_recorded_translation_is_accepted(self):
        with tempfile.TemporaryDirectory(dir=Path('.tmp').resolve()) as directory:
            root = Path(directory)
            (root / 'translation').mkdir()
            (root / 'example.py').write_bytes(b'English explanatory text')
            original = hashlib.sha256(b'original explanatory text').hexdigest()
            translated = hashlib.sha256(b'English explanatory text').hexdigest()
            self.assertFalse(source_hash_matches(root, 'example.py', original))
            record = {'translated_files': {'example.py': {'original_sha256': original, 'translated_sha256': translated}}}
            (root / 'translation/manifest.json').write_text(json.dumps(record), encoding='utf-8')
            self.assertTrue(source_hash_matches(root, 'example.py', original))
            self.assertFalse(source_hash_matches(root, 'example.py', 'unrelated recorded hash'))
            (root / 'example.py').write_bytes(b'unrecorded change')
            self.assertFalse(source_hash_matches(root, 'example.py', original))


if __name__ == '__main__':
    unittest.main()
