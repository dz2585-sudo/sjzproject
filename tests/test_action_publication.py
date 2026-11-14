import hashlib,tempfile,unittest,zipfile
from pathlib import Path
from aimrl.action_publication import verify_doc,omission
from aimrl.tracking_research.io import sha

class ActionPublicationTests(unittest.TestCase):
    def test_chained_contract_passed_explicitly_by_historical_protection(self):
        from aimrl.action_publication import install,ROOT
        old,_=install();c=old.contract();expected=c['documentation_changes']['.gitignore']['original_sha256']
        result=old.verify_protected({'.gitignore':expected},ROOT,c)
        self.assertEqual(result['allowed_publication_document_changes'],['.gitignore'])
    def test_document_change_needs_original_archive_and_current_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'README.md').write_bytes(b'reviewed')
            with zipfile.ZipFile(root/'old.zip','w') as z:z.writestr('README.md',b'baseline')
            expected=hashlib.sha256(b'baseline').hexdigest()
            c=dict(documentation_changes={'README.md':dict(original_sha256=expected,published_sha256=sha(root/'README.md'))},
                original_documents_archive='old.zip',original_documents_archive_sha256=sha(root/'old.zip'))
            verify_doc('README.md',expected,root,c)
            (root/'README.md').write_bytes(b'unreviewed')
            with self.assertRaisesRegex(ValueError,'Reviewed'):verify_doc('README.md',expected,root,c)
            with self.assertRaisesRegex(ValueError,'Unapproved'):verify_doc('core.py',expected,root,c)
    def test_only_private_snapshot_can_be_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'public.txt').write_bytes(b'PUBLICATION COPY')
            entry=dict(kind='private_identity_snapshot',original_sha256='1'*64,replacement_path='public.txt',replacement_sha256=sha(root/'public.txt'))
            c=dict(omissions={'results/tracking_action_resolution/delivery_state.json':entry,'model.npz':entry})
            omission('results/tracking_action_resolution/delivery_state.json','1'*64,root,c)
            with self.assertRaisesRegex(ValueError,'Unapproved'):omission('model.npz','1'*64,root,c)
            (root/'public.txt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Redacted'):omission('results/tracking_action_resolution/delivery_state.json','1'*64,root,c)
