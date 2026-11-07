"""Monitor full known/BNN continuations every 30 s and analyze after both terminate."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import argparse
import json
import os
import subprocess
import time
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def events(run):
    path = run / 'events.jsonl'
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def unit_state(name):
    proc = subprocess.run(['systemctl', '--user', 'show', name + '.service',
                           '-p', 'ActiveState', '-p', 'SubState', '-p', 'ExecMainStatus'],
                          capture_output=True, text=True)
    return dict(line.split('=', 1) for line in proc.stdout.splitlines() if '=' in line)


def write_status(manifest):
    out, doc = Path(manifest['output_root']), Path(manifest['documentation_root'])
    states = {}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.7))
    for arm, color in [('known-dynamics', '#245db1'), ('bnn-unknown', '#d36c15')]:
        run = out / arm
        parent = manifest['resume_checkpoints'][arm]
        prefix = read(Path(parent['checkpoint']) / 'state.json')
        es = events(run)
        attempts = [v for v in es if v.get('event') == 'attempt' and v.get('attempt_id', 0) > prefix['trials']]
        physical = [v for v in attempts if v.get('outcome') in ('success', 'failure')]
        completed = prefix['trials'] + len(physical)
        successes = prefix['counts']['success'] + sum(v.get('outcome') == 'success' for v in attempts)
        failures = prefix['counts']['failure'] + sum(v.get('outcome') == 'failure' for v in attempts)
        system = unit_state(manifest['units'][arm])
        summary = read(run / 'summary.json')
        finished = summary is not None or system.get('ActiveState') in ('inactive', 'failed')
        status = summary['status'] if summary else ('running' if not finished else 'process_error')
        previous = read(Path(manifest['parent_check_root']) / arm / 'lap_metrics.json', [])
        laps = [(v['attempt_id'], v['lap_time']) for v in previous if v['completed'] and v.get('lap_time') is not None]
        laps += [(v['lap'], v['laptime']) for v in es if v.get('event') == 'lap']
        laps = sorted(dict(laps).items())  # Resume event logs already contain the old prefix.
        latest = laps[-1][1] if laps else None
        # Completed-lap statistics never use failed or incomplete trials.
        best = min(v[1] for v in laps) if laps else None
        updates = [v for v in es if v.get('event') == 'dynamics_posterior_training']
        active_generation = updates[-1]['generation'] if updates else (prefix.get('belief_generation') or 0)
        value_events = [v for v in es if v.get('event') == 'training']
        value_generation = value_events[-1]['generation'] if value_events else prefix['value_generation']
        states[arm] = dict(status=status, finished=finished, completed_trials=completed,
                           successful_laps=successes, failed_attempts=failures,
                           latest_lap_seconds=latest, best_completed_lap_seconds=best,
                           value_generation=value_generation, belief_generation=active_generation,
                           systemd=system)
        if laps:
            x, y = np.asarray(laps).T
            ax[0].plot(x, y, color=color, lw=.8, alpha=.35)
            window=min(10, len(y))
            smoothed=np.convolve(y, np.ones(window)/window, mode='valid')
            ax[0].plot(x[window-1:],smoothed,color=color,label=arm,lw=1.5)
            ax[1].plot(x,np.minimum.accumulate(y),color=color,label=arm,lw=1.5)
        failed_ids=[v['attempt_id'] for v in previous if not v['completed']]
        failed_ids += [v['attempt_id'] for v in physical if v.get('outcome') == 'failure']
        if failed_ids:
            ax[0].plot(failed_ids,np.zeros(len(failed_ids)),'rx',transform=ax[0].get_xaxis_transform(),clip_on=False)
    for panel,title in zip(ax,('A. Completed-lap time (10-trial moving mean)','B. Best completed-lap time so far')):
        panel.set(title=title,xlabel='Trial (failed trials retained)',ylabel='Lap time (s)')
        panel.axvline(10.5,color='gray',ls=':',lw=1)
        panel.grid(alpha=.2)
        panel.legend(fontsize=8)
    fig.suptitle('Figure 1. Full known-dynamics / BNN experiment: live learning curves')
    fig.tight_layout(rect=(0,0,1,.94))
    temp=doc/'full-known-bnn-live-curves.tmp.png'
    fig.savefig(temp,dpi=150)
    temp.replace(doc/'full-known-bnn-live-curves.png')
    plt.close(fig)
    complete=sum(v['finished'] for v in states.values())
    trials=sum(v['completed_trials'] for v in states.values())
    lines=['# Full confirmation: known-dynamics SIT-LMPC and BNN',
           f'Complete/total: **{complete}/2 runs finished; {trials}/600 total attempts recorded**','',
           '## Main findings','',
           f"- Target: **300 total attempts per arm**, continuing each recorded trial-10 checkpoint. Shared plant: friction **{manifest['plant']['friction']:.6f}**, mass ratio **{manifest['plant']['mass_ratio']:.6f}**.",
           '- The known-dynamics run stops at its first physical failure. BNN trials permit retries and use the configured conservative chance fallback. A stopped arm is a completed execution outcome, not proof of 300 successful laps.',
           '- Frozen source and resolved configs preserve 1,024 action samples, 64 BNN function samples and 300-epoch NF value fitting. Both jobs share the local RTX 4090; execution speed is not a comparison metric.',
           '', '**Table 1. Current execution and learning state.**','',
           '| Arm | Status | Recorded / target trials | Success / failure | Latest / best completed lap (s) | Value / belief generation |',
           '|---|---|---:|---:|---:|---:|']
    for arm,v in states.items():
        latest=f"{v['latest_lap_seconds']:.2f}" if v['latest_lap_seconds'] is not None else '—'
        best=f"{v['best_completed_lap_seconds']:.2f}" if v['best_completed_lap_seconds'] is not None else '—'
        lines.append(f"| {arm} | {v['status']} | {v['completed_trials']} / 300 | {v['successful_laps']} / {v['failed_attempts']} | {latest} / {best} | {v['value_generation']} / {v['belief_generation']} |")
    lines += ['', '![Figure 1. Live full-run learning curves.](full-known-bnn-live-curves.png)','',
              '**Figure 1.** The dotted line marks the trial-10 continuation boundary. Thin curves are completed-lap observations; heavy curves show moving means. Failed trials remain on the x axis with red crosses; they do not enter best-lap times.',
              '',f"Updated: **{datetime.now(ZoneInfo('America/New_York')).isoformat()}**, US Eastern. Refresh interval: **30 s**.",
              '', '- [Full-run plan](full-known-versus-bnn-confirmation-plan.md).',
              '- [Configs, checkpoint provenance and source hashes](full-experiment-manifest.json).']
    final=doc/manifest['report_filename']
    if final.exists():lines+=['',f'[Final paired results and violation/belief plots]({final.name}).']
    temp=doc/'full-known-versus-bnn-live-status.md.tmp'
    temp.write_text('\n'.join(lines)+'\n')
    temp.replace(doc/'full-known-versus-bnn-live-status.md')
    (out/'full-execution-state.json').write_text(json.dumps(states,indent=2)+'\n')
    return states


def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    m=read(args.manifest)
    ROOT=Path(m.get("project_root",ROOT))
    while True:
        started=time.monotonic()
        states=write_status(m)
        if all(v['finished'] for v in states.values()):
            if all((Path(m['output_root'])/arm/'summary.json').exists() for arm in states):
                env=os.environ.copy()
                env.update(MPLBACKEND='Agg',JAX_PLATFORMS='cpu')
                with (Path(m['output_root'])/'final-analysis.log').open('a') as log:
                    subprocess.run([str(ROOT/'.venv/bin/python'),m['analysis_script'],'--manifest',str(args.manifest)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                write_status(m)
            return
        time.sleep(max(1,30-(time.monotonic()-started)))


if __name__=='__main__':
    try:
        main()
    except BaseException:
        Path(__file__).with_name('full-monitor-error.log').write_text(traceback.format_exc())
        raise
