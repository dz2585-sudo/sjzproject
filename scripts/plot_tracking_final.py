"""Clarify final-figure budgets from saved records without changing frozen training code."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from aimrl.mpl_setup import configure_mpl
from aimrl.tracking_research.io import read_json, save_json, sha
from aimrl.tracking_research.render import COLORS, LABELS, seed_values
from aimrl.tracking_research.workflow import METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/tracking_improvement')
    args = parser.parse_args()
    output = ROOT / args.output
    source = output / 'evaluation/final/summary.json'
    figure = output / 'render/final_comparison.png'
    before = sha(figure)
    data = read_json(source)
    configure_mpl()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields = ('mean_distance','close_fraction','far_fraction','hits_per_episode','pooled_shot_accuracy','mean_base_return')
    units = ('Mean L1 distance (cells)','Close fraction (d <= 1)','Far fraction (d > 5)','Hits / 300 ticks','Pooled accuracy','Unshaped base return / window')
    fig,axes = plt.subplots(2,3,figsize=(15,8))
    for ax,field,unit in zip(axes.flat,fields,units):
        values = [seed_values(data,m,field) for m in METHODS]
        ax.bar(range(4),[v.mean() for v in values],yerr=[v.std(ddof=1) for v in values],capsize=3,color=COLORS)
        for i,v in enumerate(values): ax.scatter([i]*3,v,color='black',s=12,zorder=3)
        for ref,style in [('historical_potential','--'),('rule',':')]:
            ax.axhline(data[ref][field],linestyle=style,color='#555555' if ref=='rule' else '#a35c00',label=LABELS[ref])
        ax.set_xticks(range(4),['Abs Base\n5M','Abs Potential\n5M','Rel Base\n5M','Rel Potential\n5M'])
        ax.set_ylabel(unit); ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Independent final test: 500 paired windows | new models: 5M interactions each\nBars = three-seed means; error bars = sample SD; dots = all seeds | archived 20M model is a cross-budget reference')
    fig.tight_layout(); fig.savefig(figure,dpi=150); plt.close(fig)
    save_json(output/'render/figure_label_review.json',{'reason':'Post-training figure review clarified the new-model 5M budget; numerical data unchanged.',
        'script_sha256':sha(Path(__file__)),'source_records_sha256':sha(source),'previous_figure_sha256':before,
        'figure_sha256':sha(figure),'frozen_training_sources_or_protocol_modified':False})
    print('Final comparison regenerated from saved results with explicit5M/20M budget labels.')


if __name__ == '__main__': main()
