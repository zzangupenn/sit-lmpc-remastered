"""Run explicit known/unknown-dynamics checks sequentially with 30 s Markdown updates."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import argparse
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def events(run):
    path = run / 'events.jsonl'
    if not path.exists():
        return []
    values = []
    for line in path.read_text().splitlines():
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return values


def status(manifest, states):
    doc, output = Path(manifest['documentation_root']), Path(manifest['output_root'])
    (output / 'execution-state.json').write_text(json.dumps(states, indent=2) + '\n')
    done = sum(s.get('finished', False) for s in states.values())
    plant = manifest['plant']
    lines = ['# Known-dynamics and unknown-plant BNN check runs',
             f'Complete/total: **{done}/2 check runs**', '', '## Main findings', '',
             f"- Shared ST plant: friction **{plant['friction']:.6f}**, mass ratio **{plant['mass_ratio']:.6f}**. The analytical arm knows these parameters; the BNN arm does not receive them as predictor inputs.",
             '- Normal 1,024-action / 10-step controller; 64 coherent BNN functions, calibrated posterior updates and chance constraints. Both arms retain 300-epoch NF value training.',
             '- These are ten-attempt functional checks on one sampled plant, not a multi-seed robustness evaluation.', '',
             '**Table 1. Live arm status.**', '',
             '| Arm | Status | Successful / failed attempts | Latest completed lap (s) | Results |',
             '|---|---|---:|---:|---|']
    for arm in ('known-dynamics', 'bnn-unknown'):
        run = output / arm
        es = events(run)
        attempts = [v for v in es if v.get('event') == 'attempt']
        success = sum(v.get('outcome') == 'success' for v in attempts)
        failed = sum(v.get('outcome') == 'failure' for v in attempts)
        laps = [v for v in es if v.get('event') == 'lap']
        duration = laps[-1].get('laptime') if laps else None
        duration_text = f'{duration:.2f}' if duration is not None else '—'
        state = states[arm]
        label = state['status']
        last = es[-1].get('event') if es else None
        if state['status'] == 'running' and last:
            label += f' ({last})'
        lines.append(f'| {arm} | {label} | {success} / {failed} | {duration_text} | [{arm}]({run}) |')
    lines += ['', f"Updated: **{datetime.now(ZoneInfo('America/New_York')).isoformat()}**, US Eastern; refresh every 30 s.", '',
              '- [Run plan](known-versus-bnn-check-plan.md).', '- [Parameters, seed and config hashes](check-run-manifest.json).']
    final = doc / 'known-versus-bnn-sampled-plant-results.md'
    if final.exists():
        lines += ['', '[Completed results and figures](known-versus-bnn-sampled-plant-results.md).']
    target = doc / 'known-versus-bnn-live-status.md'
    temp = target.with_suffix('.md.tmp')
    temp.write_text('\n'.join(lines) + '\n')
    temp.replace(target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    args = p.parse_args()
    manifest = read(args.manifest)
    output = Path(manifest['output_root'])
    states = {key: {'status': 'pending', 'finished': False} for key in ('known-dynamics', 'bnn-unknown')}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    env.pop('JAX_PLATFORMS', None)
    status(manifest, states)
    for arm in states:
        states[arm]['status'] = 'running'
        status(manifest, states)
        command = [str(ROOT / '.venv/bin/sit-racecar'), '--config', str(Path(manifest['config_root']) / f'{arm}.yaml')]
        started = time.monotonic()
        with (output / f'{arm}.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                status(manifest, states)
                time.sleep(30)
        summary = read(output / arm / 'summary.json', {})
        states[arm] = dict(status=summary.get('status', 'process_error'), finished=True,
                           return_code=child.returncode, elapsed_seconds=time.monotonic() - started)
        (output / 'execution-state.json').write_text(json.dumps(states, indent=2) + '\n')
        status(manifest, states)
    command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/analyze_known_bnn_check.py'),
               '--manifest', str(args.manifest)]
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    status(manifest, states)


if __name__ == '__main__':
    main()
