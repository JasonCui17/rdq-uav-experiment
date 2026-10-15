#!/usr/bin/env python3
"""Ablate selector weights on a frozen physical track bank; no GT in choice."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_tracking import metrics, write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    queries = [json.loads(line) for line in (args.source / 'per_query.jsonl').open()]
    method = 'multi_candidate_kf_select'
    # Associate rounded datetime timestamps with the original query, tolerance
    # < 1 microsecond; never use target coordinates to match records.
    by_time = {(q['sequence_id'], round(q['query_time'], 5)): q['sample_id'] for q in queries}
    states = defaultdict(dict)
    for line in (args.source / 'all_track_history.jsonl').open():
        row = json.loads(line)
        if row['method'] == method:
            key = by_time.get((row['sequence_id'], round(row['timestamp'], 5)))
            if key is None:
                # At decimal rounding boundaries use the actual nearest time.
                matches = [q for q in queries if q['sequence_id'] == row['sequence_id']
                           and abs(q['query_time'] - row['timestamp']) < 1e-6]
                if len(matches) != 1:
                    raise ValueError('Ambiguous track timestamp')
                key = matches[0]['sample_id']
            states[key][row['track_id']] = row
    scores = defaultdict(dict)
    for line in (args.source / 'track_scores.jsonl').open():
        row = json.loads(line)
        if row['method'] == method:
            scores[row['sample_id']][row['track_id']] = row
    cfg = json.loads((args.source / 'multi_tracker_config.json').read_text())['selection']
    variants = {'frozen_original': {}, 'no_support': {'support_weight': 0.},
                'score_only': {k: 0. for k in ('support_weight', 'recency_weight',
                                              'innovation_weight', 'motion_weight',
                                              'background_weight')}}
    (args.output / 'protocol.json').write_text(json.dumps({
        'source': str(args.source.resolve()),
        'configs': {name: {**cfg, **overrides} for name, overrides in variants.items()},
        'GT_used_for_choice': False, 'physical_bank_frozen': True}, indent=2))
    summary = {}
    for name, overrides in variants.items():
        config = {**cfg, **overrides}
        current = challenger = sequence = None
        streak = 0
        rows = []
        for q in queries:
            if sequence != q['sequence_id']:
                current = challenger = None
                streak = 0
                sequence = q['sequence_id']
            ranked = {}
            for number, s in scores[q['sample_id']].items():
                if not s['eligible_for_output']:
                    continue
                total = sum(config[w] * (s[v] or 0.) for w, v in (
                    ('score_weight', 'target_score'), ('support_weight', 'observation_support'),
                    ('recency_weight', 'recency'), ('innovation_weight', 'innovation_consistency'),
                    ('motion_weight', 'reliable_motion_evidence')))
                penalty = config['background_weight'] * (s['static_background_hint'] or 0.)
                total -= min(penalty, max(0., total - config['min_selection_score']))
                if total >= config['min_selection_score']:
                    ranked[number] = total
            if current not in ranked:
                current = challenger = None
                streak = 0
            if ranked:
                winner = max(ranked, key=lambda n: (ranked[n], -n))
                if current is None:
                    current = winner
                elif winner == current or ranked[winner] < ranked[current] + config['switch_margin']:
                    challenger, streak = None, 0
                else:
                    streak = streak + 1 if challenger == winner else 1
                    challenger = winner
                    if streak >= config['switch_consecutive']:
                        current, challenger, streak = winner, None, 0
            xyz = states[q['sample_id']][current]['xyz'] if current is not None else None
            error = float(np.linalg.norm(np.array(xyz) - q['gt_xyz'])) if xyz is not None and q['has_gt'] else None
            rows.append({**{k: q[k] for k in ('sample_id', 'sequence_id', 'query_time', 'has_gt')},
                         name + '_error_m': error, 'track_id': current, 'xyz': xyz})
        summary[name] = metrics(rows, name)
        write_csv(args.output / (name + '.csv'), rows)
        if name == 'frozen_original':
            actual = [r[name + '_error_m'] for r in rows]
            expected = [q[method + '_error_m'] for q in queries]
            if actual != expected:
                raise AssertionError('Frozen replay must reproduce original outputs exactly')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
