"""Fail-closed sequence identity and causal-history invariants (no GT access)."""
import math
import torch


def assert_temporal_clip_integrity(queries,clip_index=None,require_events=False):
    """Check one unpadded clip; metadata-only requests may omit event metadata.

    Loaded queries and prechecks must use require_events=True. Errors include
    full clip identities/times so duplicate timestamps are never silently fixed.
    """
    seq=[q.get('sequence_id') for q in queries]
    times=[q.get('query_time') for q in queries]
    ids=[q.get('sample_id') for q in queries]
    context=f'clip_index={clip_index}, sequence_ids={seq}, query_times={times}, sample_ids={ids}'
    def check(condition,message):
        if not condition:raise AssertionError(f'{message}: {context}')
    check(bool(queries),'empty_temporal_clip')
    check(all(isinstance(s,str) and bool(s.strip()) for s in seq),'empty_sequence_id')
    check(len(set(seq))==1,'cross_sequence_clip')
    check(all(t is not None and math.isfinite(float(t)) for t in times),'invalid_query_timestamp')
    check(len(set(times))==len(times),'duplicate_query_timestamp')
    check(all(b>a for a,b in zip(times,times[1:])),'unsorted_query_timestamps')
    check(all(isinstance(s,str) and bool(s) for s in ids) and len(set(ids))==len(ids),'duplicate_or_missing_sample_identity')
    for q in queries:
        if require_events or 'event_timestamps' in q:
            check('event_sequence_ids' in q and 'event_timestamps' in q,'missing_event_identity')
            ts=q['event_timestamps'];es=q['event_sequence_ids']
            check(len(es)==len(ts)==q['event_count'],'event_metadata_length_mismatch')
            check(all(s==q['sequence_id'] for s in es),'event_sequence_mismatch')
            check(all(math.isfinite(float(t)) and t<=q['query_time'] for t in ts),'future_or_invalid_event')
        if 'delta_t' in q:
            check(bool(torch.isfinite(q['delta_t']).all() & (q['delta_t']<=0).all()),'future_or_invalid_delta_t')


def assert_temporal_batch_integrity(batch):
    """Verify flat query identities before restoring [B,T] by explicit indices.

    Times are not identities. Every (clip_batch_index,clip_position) occurs once;
    each valid clip is one sequence with strictly increasing query times.
    """
    valid=batch['query_valid_mask'];B,T=valid.shape;n=int(batch['num_samples'])
    cb=batch['clip_batch_index'];cp=batch['clip_position']
    if n!=B*T or cb.shape!=(n,) or cp.shape!=(n,):raise AssertionError('Invalid flattened clip layout')
    key=cb*T+cp
    if bool(((cb<0)|(cb>=B)|(cp<0)|(cp>=T)).any()) or not torch.equal(torch.sort(key).values,torch.arange(n,device=key.device)):
        raise AssertionError('Duplicate/out-of-range flattened query identity')
    if len(batch['sequence_id'])!=n or len(batch['sample_id'])!=n:raise AssertionError('Missing query identities')
    if not torch.equal(batch['query_time'],batch['query_time_clip'][cb,cp]):raise AssertionError('Query time/layout mismatch')
    for b in range(B):
        ids=torch.nonzero((cb==b)&valid[cb,cp]).flatten()
        ids=ids[torch.argsort(cp[ids])].tolist()
        queries=[dict(sequence_id=batch['sequence_id'][i],sample_id=batch['sample_id'][i],query_time=float(batch['query_time'][i])) for i in ids]
        assert_temporal_clip_integrity(queries,clip_index=b)
