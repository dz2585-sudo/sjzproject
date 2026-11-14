"""Publication exceptions are explicit, narrow and fail closed."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from aimrl.latest_publication import verify_protected,verify_document,verify_omission,check_links,verify_release_inputs
from aimrl.tracking_research.io import sha


class LatestPublicationTests(unittest.TestCase):
    def test_actual_release_inputs_and_relative_links(self):
        self.assertFalse(verify_release_inputs()['changed_experiment_core_files'])
        self.assertGreater(check_links()['relative_links_checked'],30)

    def test_unlisted_mutation_and_missing_core_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'core.py';p.write_bytes(b'changed')
            c=dict(documentation_changes={},omissions={})
            with self.assertRaisesRegex(ValueError,'Unapproved'):verify_protected({'core.py':'0'*64},root,c)
            with self.assertRaisesRegex(ValueError,'Undocumented'):verify_protected({'model.npz':'0'*64},root,c)

    def test_document_requires_both_original_archive_and_reviewed_current_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);original=b'original doc';(root/'README.md').write_bytes(b'published doc')
            with zipfile.ZipFile(root/'original.zip','w') as z:z.writestr('README.md',original)
            digest=hashlib.sha256(original).hexdigest()
            c=dict(documentation_changes={'README.md':dict(original_sha256=digest,published_sha256=sha(root/'README.md'))},
                   original_documents_archive='original.zip',original_documents_archive_sha256=sha(root/'original.zip'),omissions={})
            self.assertEqual(verify_protected({'README.md':digest},root,c)['allowed_publication_document_changes'],['README.md'])
            (root/'README.md').write_bytes(b'not reviewed')
            with self.assertRaisesRegex(ValueError,'after review'):verify_document('README.md',digest,root,c)

    def test_omission_cannot_hide_model_and_replacement_digest_is_required(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'copy.txt').write_bytes(b'PUBLICATION COPY')
            entry=dict(kind='redacted_failure_transcript',original_sha256='1'*64,replacement_path='copy.txt',replacement_sha256=sha(root/'copy.txt'))
            c=dict(omissions={'results/x/execution/failure.txt':entry})
            verify_omission('results/x/execution/failure.txt','1'*64,root,c)
            (root/'copy.txt').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'replacement differs'):verify_omission('results/x/execution/failure.txt','1'*64,root,c)
            c=dict(omissions={'results/x/model.npz':dict(kind='native_intermediate',original_sha256='1'*64)})
            with self.assertRaisesRegex(ValueError,'not a native'):verify_omission('results/x/model.npz','1'*64,root,c)
