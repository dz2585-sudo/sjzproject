"""Run the previous canonical verifier with receipts redirected to new outputs.

Only receipt destinations change. Historical source/model hashes, expected
digests and all original checks remain intact. Never write into protected work.
"""
import argparse
from pathlib import Path

from ..tracking_research.io import save_json
from .core import PREVIOUS
from .reference import require


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    target=args.output.resolve();require(not target.is_relative_to(PREVIOUS),'Do not overwrite previous results')
    from ..tracking_boundary_stability import verify_artifacts as canonical
    from ..tracking_boundary_stability import verify as frozen
    require(canonical.OUT==PREVIOUS and frozen.OUT==PREVIOUS,'Unexpected previous output directory')
    def redirected(path,value):
        path=Path(path)
        require(path.parent==PREVIOUS and path.name in ('verification_python.json','verification_optimized.json'),'Unexpected previous-study write')
        save_json(target,value)
    frozen.save_json=redirected;canonical.save_json=redirected
    canonical.main()


if __name__=='__main__':main()
