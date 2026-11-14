"""Canonical-state adapter for the frozen artifact verifier.

The frozen verifier passes a NumPy int16 initial-state array to historical
window_metrics. target_hash historically expects a tuple: bytes(tuple) encodes
one byte per value, while bytes(int16_array) emits two bytes per value. This
adapter restores the actual evaluation call contract. It changes no expected
digest, physical trace, metric definition, protocol, model, or frozen source.
Every original check still executes, including exact target digest comparisons.
"""
import json
import numpy as np

from ..tracking_research.io import save_json,sha
from . import verify as frozen
from .core import Encoder,OUT,config,require

_window_metrics=frozen.window_metrics


def canonical_window_metrics(initial,ticks,*args,**kwargs):
    s=np.asarray(initial)
    require(s.shape==(6,) and s.dtype.kind in 'iu','Invalid saved initial-state array')
    physical=tuple(map(int,s))
    Encoder('R').index(physical)
    return _window_metrics(physical,ticks,*args,**kwargs)


def main():
    c=config()
    # Test the precise serialization repair against an existing saved digest,
    # independently of the later exhaustive original verification.
    path=OUT/'evaluation/final/R-Fixed_seed60_rollouts.npz'
    from ..tracking_research.io import read_csv
    with np.load(path,allow_pickle=False) as z:
        initial,ticks=z['initial'][0],z['ticks'][0]
        canonical=canonical_window_metrics(initial,ticks)
        original_tuple=_window_metrics(tuple(map(int,initial)),ticks)
        wrong_array=_window_metrics(initial,ticks)
        require(canonical==original_tuple,'Canonical tuple adapter changed window metrics')
        require(canonical['target_sha256']!=wrong_array['target_sha256'],'Expected tuple/array serialization distinction absent')
        saved=read_csv(OUT/'evaluation/final/R-Fixed_seed60_guard_windows.csv')[0]
        require(canonical['target_sha256']==saved['target_sha256'],'Canonical target digest does not match original evaluation')
    frozen.window_metrics=canonical_window_metrics
    result=frozen.verify(c)
    result['validation_adapter']=dict(source='aimrl/tracking_boundary_stability/verify_artifacts.py',source_sha256=sha(__file__),
                 frozen_verifier_sha256=sha(frozen.__file__),
                 repair='Normalize saved int16 initial state to the same six-Python-int tuple used by actual evaluation before bytes(initial[2:]) hashing.',
                 canonical_contract_test_passed=True,expected_digests_changed=False,original_checks_disabled=0)
    destination=OUT/('verification_optimized.json' if not __debug__ else 'verification_python.json')
    save_json(destination,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
