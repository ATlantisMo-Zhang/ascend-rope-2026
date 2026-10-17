"""Verify archived identities and historical comparisons, without CANN or network."""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise SystemExit('FAIL: ' + message)


def score(times, reference):
    return sum(100 / (1 + math.log(times[k] / reference[k], 1.5)) for k in reference) / len(reference)


def main():
    manifest = load('evidence/manifest.json')
    for item in manifest['files']:
        p = ROOT / item['path']
        require(p.is_file(), 'missing ' + item['path'])
        data = p.read_bytes()
        require(len(data) == item['bytes'], 'byte count ' + item['path'])
        require(hashlib.sha256(data).hexdigest() == item['sha256'], 'SHA256 ' + item['path'])
    b = load('evidence/platform/509159.json')
    final = load('evidence/platform/721646.json')
    q = final['latest_query']
    require(final['training_only'] is True, 'training evidence type')
    require(q['record_number'] == 721646 and q['status'] == 'Pass', 'final record')
    require(final['passed_testcases'] == 15 and len(q['result']) == 15, 'final 15 points')
    require(all(x['testcase_status'] == 'Pass' for x in q['result']), 'all final statuses')
    require(len(b['cases']) == 15 and all(x['status'] == 'Pass' for x in b['cases']), 'baseline 15 points')
    for name, digest in [('final/kernel.asc', q['kernel']['sha256']),
                         ('baselines/v3/kernel.asc', b['source_sha256'])]:
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name + ' binding')
    with (ROOT / 'evidence/baseline-vs-final.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 15, 'CSV point count')
    for idx, (row, old, new) in enumerate(zip(rows, b['cases'], q['result']), 1):
        value, unit = old['time_display'].split()
        require(unit in ('us', 'ms'), 'historical time unit')
        baseline_us = float(value) * (1000 if unit == 'ms' else 1)
        require(baseline_us == old['time_us'] == float(row['baseline_509159_us']), 'baseline unit conversion')
        require(int(row['point']) == old['index'] == idx, 'point order')
        require(row['testcase_id'] == new['testcase_id'], 'testcase identity')
        require(float(row['final_721646_us']) == new['time'], 'final time')
        require(abs(float(row['speedup']) - baseline_us/new['time']) < 1e-6, 'speedup')
        require(abs(float(row['latency_reduction_percent']) - 100*(1-new['time']/baseline_us)) < 1e-6, 'reduction')
    s = load('evidence/selection.json')
    require(len(s['records']) == 163, 'selection record count')
    for row in s['records']:
        require(set(row['times']) == set(s['reference']), 'selection 15-point identities')
        require(abs(score(row['times'], s['reference']) - row['score']) < 1e-8, 'recomputed score')
    ordered = sorted(s['records'], key=lambda row: row['score'], reverse=True)
    require([r['record'] for r in ordered[:2]] == [686549, 680947], 'top selection records')
    require(ordered[0]['sha256'] == ordered[1]['sha256'] == q['kernel']['sha256'], 'selected source')
    for p in sorted((ROOT / 'evidence/platform').glob('*.json')):
        if p.stem == '509159':
            continue
        data = json.loads(p.read_text(encoding='utf-8'))
        query = data['latest_query']
        require(query['record_number'] == int(p.stem), 'record filename')
        require(data['training_only'] and query['status'] == 'Pass' and len(query['result']) == 15, 'platform record scope')
        source = ROOT / ('final/kernel.asc' if query['kernel']['sha256'] == q['kernel']['sha256'] else f'evidence/sources/{p.stem}.asc')
        if source.is_file():
            require(hashlib.sha256(source.read_bytes()).hexdigest() == query['kernel']['sha256'], 'record/source hash')
    snap = load('evidence/ranking-snapshot.json')
    require(snap['own_record'] == 721646 and snap['own_rank'] == 59 and snap['own_score'] == 47.54, 'snapshot')
    require(snap['real_rank10']['record'] == 720247 and len(snap['real_rank10']['result']) == 15, 'real opponent')
    latest = load('evidence/resubmission-20261015.json')
    recent = load('evidence/platform/826799.json')
    require(latest['record'] == 826799 and latest['rank'] == 69 and latest['score'] == 45.28, 'latest historical snapshot')
    require(latest['source_sha256'] == recent['latest_query']['kernel']['sha256'] == q['kernel']['sha256'], 'same source resubmission')
    require(latest['no_new_optimization'] and latest['training_only'], 'resubmission scope')
    print(f"PASS: {len(manifest['files'])} file identities; 15-point historical table; 163 rescored records; source bindings.")
    print('Scope: archive consistency only. No Ascend compilation, NPU execution, or current leaderboard query.')


if __name__ == '__main__':
    main()
