"""Regenerate the published figures from the compact CSV results; no simulator required."""
import argparse
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = {'known-dynamics': '#245db1', 'bnn-unknown': '#d36c15'}
LABELS = {'known-dynamics': 'Known ST / SIT-LMPC', 'bnn-unknown': 'Unknown ST / BNN + chance'}


def records(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    value = row.get(key)
    return float(value) if value not in (None, '') else np.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'figures')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((HERE / 'metrics.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))
    panels = [('lap_time', 'A. Completed-lap time', 'Time (s)'),
              ('mean_speed', 'B. Measured trial mean speed', 'Speed (m/s)'),
              ('actual_boundary_max', 'C. Measured physical boundary violation', 'Violation (m)'),
              ('safe_set_max', 'D. Nominal terminal safe-set violation', 'Mixed-coordinate hull distance')]
    for arm, color in COLORS.items():
        rows = records(HERE / 'data' / f'{arm}-laps.csv')
        x = [number(row, 'attempt_id') for row in rows]
        for panel, (key, title, ylabel) in zip(axes.ravel(), panels):
            y = [number(row, key) if key != 'lap_time' or row['completed'] == 'True' else np.nan for row in rows]
            panel.plot(x, y, color=color, lw=1, label=LABELS[arm])
            absent = ~np.isfinite(y)
            if np.any(absent):
                panel.plot(np.asarray(x)[absent], np.zeros(absent.sum()), 'rx', transform=panel.get_xaxis_transform(), clip_on=False)
            panel.set(title=title, ylabel=ylabel, xlabel='Trial (including failures)')
    for panel in axes.ravel():
        panel.grid(alpha=.2)
        panel.legend(fontsize=8)
        panel.axvline(5.5, color='gray', ls=':', lw=1)
        panel.set_xticks([1, 50, 100, 150, 200, 250, 300])
    plant = metrics['plant']
    fig.suptitle(f"Figure 1. Same sampled plant: friction={plant['friction']:.4f}, mass ratio={plant['mass_ratio']:.4f}")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(args.output / 'known-vs-bnn-learning-and-violations.png', dpi=160)
    plt.close(fig)

    rows = records(HERE / 'data/bnn-belief-updates.csv')
    x = [number(row, 'attempt_id') for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.7))
    panels = [('likelihood_power', 'A. Published evidence power', 'Power'),
              ('latent_covariance_trace_ratio', 'B. Gaussian covariance trace / prior', 'Trace ratio'),
              ('xy_rmse_10', 'C. Held-out 10-step mean-prediction error', 'XY RMSE (m)'),
              ('coverage_10', 'D. Held-out coherent path-tube coverage', 'Coverage')]
    for panel, (key, title, ylabel) in zip(axes.ravel(), panels):
        panel.plot(x, [number(row, key) for row in rows], lw=1.1)
        panel.set(title=title, ylabel=ylabel, xlabel='Attempt-boundary update considered')
        panel.grid(alpha=.2)
    axes[0, 0].set_yscale('log')
    axes[1, 1].axhline(.95, color='gray', ls='--', label='Calibration target')
    axes[1, 1].set_ylim(-.02, 1.02)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle('Figure 2. BNN belief learning and held-out diagnostics')
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(args.output / 'bnn-belief-updates-and-prediction.png', dpi=160)
    plt.close(fig)
    print(f'[figures_saved] {args.output.resolve()}')


if __name__ == '__main__':
    main()
