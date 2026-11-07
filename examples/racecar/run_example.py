"""Run a portable known-ST or BNN example, including the recorded trial-10 resume boundary."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo
import json
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def phase_config(controller, directory, max_laps, *, resume=None, render=False):
    """Use the fully specified example settings, independent of the default preset."""
    config = yaml.safe_load((HERE / 'configs' / f'{controller}.yaml').read_text())
    config['simulation']['max_laps'] = max_laps
    config['output'].update(directory=str(directory), resume=str(resume) if resume else None)
    config['diagnostics']['render'] = render
    if resume and controller == 'bnn-unknown':
        config['planner']['dynamics']['checkpoint'] = None  # The checkpoint bundles its population.
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', choices=('known-dynamics', 'bnn-unknown', 'both'), default='both')
    parser.add_argument('--max-laps', type=int, default=300)
    parser.add_argument('--prefix-laps', type=int, default=10)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    if args.max_laps < 1 or args.prefix_laps < 1:
        parser.error('--max-laps and --prefix-laps must be positive')
    stamp = datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d-%H%M%S')
    root = (args.output_root or ROOT / 'results/custom1' / f'racecar-example-{stamp}').expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        parser.error(f'Refusing to overwrite nonempty output root: {root}')
    root.mkdir(parents=True, exist_ok=True)
    arms = ('known-dynamics', 'bnn-unknown') if args.controller == 'both' else (args.controller,)
    for arm in arms:
        for name, target, source in [('prefix', min(args.prefix_laps, args.max_laps), None),
                                     ('continuation', args.max_laps, 'prefix')]:
            if name == 'continuation' and args.max_laps <= args.prefix_laps:
                continue
            resume = None
            if source:
                summary = json.loads((root / arm / source / 'summary.json').read_text())
                if summary['status'] != 'lap_limit':
                    print(f'[example_stop] {arm}: prefix ended with {summary["status"]}')
                    return 2
                resume = root / arm / source / 'checkpoints' / f'trial-{args.prefix_laps:06d}'
                if not resume.is_dir():
                    raise FileNotFoundError(f'Expected final prefix checkpoint: {resume}')
            config = phase_config(arm, root / arm / name, target, resume=resume, render=args.render)
            path = root / f'{arm}-{name}.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            print(f'[example_run] {arm} {name} | target={target} total attempts | config={path}', flush=True)
            result = subprocess.run([sys.executable, '-m', 'sit_lmpc.cli.racecar', '--config', str(path)], cwd=ROOT)
            if result.returncode:
                return result.returncode
    print(f'[example_saved] {root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
