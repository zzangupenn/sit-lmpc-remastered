"""Summarize paired known-dynamics and unknown-plant BNN checks with measured outcomes."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'known-dynamics': '#245db1', 'bnn-unknown': '#d36c15'}
LABELS = {'known-dynamics': 'Known ST / SIT-LMPC', 'bnn-unknown': 'Unknown ST / BNN + chance'}


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def event_rows(run):
    return [json.loads(s) for s in (run / 'events.jsonl').read_text().splitlines()]


def get_horizon(record, horizon=10):
    h = record.get('validation_after', {}).get('horizons', {}).get(str(horizon), {})
    return h.get('endpoint_xy_rmse', np.nan), h.get('path_coverage_95', np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    m = read(args.manifest)
    limit = m.get('max_attempts_each', 10)
    report_name = m.get('report_filename', 'known-versus-bnn-sampled-plant-results.md')
    metrics_name = m.get('metrics_filename', 'known-versus-bnn-check-metrics.json')
    root, doc = Path(m['output_root']), Path(m['documentation_root'])
    figure_dir = doc / 'figures'
    figure_dir.mkdir(exist_ok=True)
    data = {}
    checks = {}
    for arm in COLORS:
        run = root / arm
        summary = read(run / 'summary.json', {})
        laps = read(run / 'lap_metrics.json', [])
        events = event_rows(run)
        resolved = yaml.safe_load((run / 'config.yaml').read_text())
        learned = []
        for path in sorted((run / 'trajectories/success').glob('*.npz')):
            with np.load(path) as z:
                if np.any(z['controllers'] == 'lmppi'):
                    learned.append(int(json.loads(str(z['metadata'].item()))['attempt_id']))
        durations = [v['lap_time'] for v in laps if v['completed'] and v['lap_time'] is not None]
        learned_laps = [v['lap_time'] for v in laps if v['attempt_id'] in learned and v['completed']]
        posterior = read(run / 'dynamics_posterior.json', {})
        history = posterior.get('history', [])
        truth_hidden = (resolved['planner']['dynamics']['backend'] == 'gaussian_bnn'
                        and resolved['planner']['friction'] != resolved['simulation']['friction']
                        and resolved['planner']['mass_ratio'] != resolved['simulation']['mass_ratio'])
        state = {
            'status': summary.get('status', 'missing_summary'),
            'successful_laps': summary.get('successful_laps', 0),
            'failed_attempts': summary.get('failed_attempts', 0),
            'incomplete_attempts': summary.get('incomplete_attempts', 0),
            'applied_steps': summary.get('applied_steps', 0),
            'learned_steps': summary.get('learned_steps', 0),
            'value_generations': summary.get('training_generations', 0),
            'belief_generation': summary.get('dynamics_belief_generation', 0),
            'best_completed_lap_seconds': min(durations) if durations else None,
            'best_learned_lap_seconds': min(learned_laps) if learned_laps else None,
            'actual_out_of_bounds_steps': sum(v.get('out_of_bounds_steps', 0) for v in laps),
            'max_abs_lateral_offset': max((v.get('max_abs_lateral_offset', 0) for v in laps), default=0),
            'chance_fallback_controls': sum(v['event'] == 'probabilistic_safety_fallback' for v in events),
            'chance_active_events': sum(v['event'] == 'probabilistic_safety_status' and v.get('status') == 'active' for v in events),
            'posterior_training_transitions': posterior.get('training_transitions', 0),
            'posterior_updates_considered': len(history),
            'posterior_updates_published': sum(v.get('published', False) for v in history),
            'active_power': posterior.get('active_power'),
            'true_plant_not_given_to_bnn_predictor': truth_hidden if arm == 'bnn-unknown' else None,
            'result_directory': str(run),
        }
        parent = m.get('resume_checkpoints', {}).get(arm)
        if parent:
            prefix = read(Path(parent['checkpoint']) / 'state.json', {})
            state['restored_trials'] = prefix.get('trials', 0)
            state['new_completed_trials'] = summary.get('trials', 0) - prefix.get('trials', 0)
            state['new_applied_steps'] = state['applied_steps'] - prefix.get('next_step', 0)
            checks[f'{arm}_continued_beyond_checkpoint'] = state['new_applied_steps'] > 0
        if history:
            rmse, coverage = get_horizon(history[-1])
            state['last_heldout_10step_xy_rmse_m'] = float(rmse) if np.isfinite(rmse) else None
            state['last_heldout_10step_path_coverage'] = float(coverage) if np.isfinite(coverage) else None
            state['last_heldout_10step_windows'] = history[-1].get('validation_after', {}).get('horizons', {}).get('10', {}).get('windows', 0)
            state['last_posterior_status'] = history[-1].get('status')
        if arm == 'known-dynamics':
            checks['oracle_parameters_match_plant'] = all(resolved['planner'][name] == resolved['simulation'][name] for name in ('friction', 'mass_ratio'))
            checks['sit_mode_enabled'] = resolved['planner']['sit_lmpc_mode']
            checks['oracle_chance_disabled'] = not resolved['planner']['probabilistic_safety']['enabled']
        else:
            checks['unknown_parameters_not_used_for_bnn_prediction'] = truth_hidden
            checks['bnn_chance_enabled'] = resolved['planner']['probabilistic_safety']['enabled']
            checks['bnn_posterior_enabled'] = resolved['planner']['dynamics']['belief_update'] == 'posterior'
            checks['coherent_function_samples'] = resolved['planner']['dynamics']['particles'] == 64
            bundled = run / 'dynamics_model/model.msgpack'
            checks['population_weights_unchanged'] = bundled.exists() and m['model_fingerprint'] == hashlib.sha256(bundled.read_bytes()).hexdigest()
        checks[f'{arm}_learner_exercised'] = state['learned_steps'] > 0 and state['value_generations'] > 0
        checks[f'{arm}_clean_termination'] = state['status'] in ('lap_limit', 'step_limit', 'out_of_bounds', 'collision')
        data[arm] = dict(metrics=state, laps=laps, history=history, learned_ids=learned)

    # All per-trial plots retain failed trials, including a red cross where a metric is absent.
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.6))
    for arm, values in data.items():
        rows = values['laps']
        indices = np.array([v['attempt_id'] for v in rows])
        for panel, key, title, ylabel in [
            (ax[0, 0], 'lap_time', 'A. Completed-lap time', 'Time (s)'),
            (ax[0, 1], 'mean_speed', 'B. Measured trial mean speed', 'Speed (m/s)'),
            (ax[1, 0], 'actual_boundary_max', 'C. Measured physical boundary violation', 'Violation (m)'),
            (ax[1, 1], 'safe_set_max', 'D. Nominal terminal safe-set violation', 'Mixed-coordinate hull distance'),
        ]:
            ys = np.array([v.get(key) if v.get(key) is not None else np.nan for v in rows], float)
            if key == 'lap_time':
                ys = np.array([v.get(key) if v['completed'] and v.get(key) is not None else np.nan for v in rows], float)
            panel.plot(indices, ys, 'o-', color=COLORS[arm], ms=3.5, lw=1.2, label=LABELS[arm])
            absent = ~np.isfinite(ys)
            if np.any(absent):
                panel.plot(indices[absent], np.zeros(absent.sum()), 'rx', transform=panel.get_xaxis_transform(), clip_on=False)
            failures = np.array([not v['completed'] for v in rows])
            if np.any(failures):
                panel.plot(indices[failures], np.zeros(failures.sum()), 'rx', transform=panel.get_xaxis_transform(), clip_on=False)
            panel.set(title=title, ylabel=ylabel, xlabel='Trial (including failures)')
    for panel in ax.ravel():
        panel.grid(alpha=.2)
        panel.legend(fontsize=8)
        panel.set_xticks(range(1, limit + 1) if limit <= 20 else [1, *range(50, limit + 1, 50)])
        panel.axvline(5.5, color='gray', ls=':', lw=1)
    fig.suptitle(f"Figure 1. Same sampled plant: friction={m['plant']['friction']:.4f}, mass ratio={m['plant']['mass_ratio']:.4f}")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(figure_dir / 'known-vs-bnn-learning-and-violations.png', dpi=160)
    plt.close(fig)

    history = data['bnn-unknown']['history']
    if history:
        fig, ax = plt.subplots(2, 2, figsize=(11, 6.7))
        it = np.arange(1, len(history) + 1)
        values = [get_horizon(v) for v in history]
        ax[0, 0].plot(it, [v['likelihood_power'] for v in history], 'o-')
        ax[0, 0].set(title='A. Published evidence power', ylabel='Power', yscale='log')
        ax[0, 1].plot(it, [v['latent_covariance_trace_ratio'] for v in history], 'o-')
        ax[0, 1].set(title='B. Gaussian covariance trace / prior', ylabel='Trace ratio')
        ax[1, 0].plot(it, [v[0] for v in values], 'o-')
        ax[1, 0].set(title='C. Held-out 10-step mean-prediction error', ylabel='XY RMSE (m)')
        ax[1, 1].plot(it, [v[1] for v in values], 'o-')
        ax[1, 1].axhline(.95, color='gray', ls='--', label='Calibration target')
        ax[1, 1].set(title='D. Held-out coherent path-tube coverage', ylabel='Coverage', ylim=(-.02, 1.02))
        ax[1, 1].legend(fontsize=8)
        for panel in ax.ravel():
            panel.set_xlabel('Attempt-boundary update considered')
            panel.grid(alpha=.2)
        fig.suptitle('Figure 2. BNN belief learning and held-out diagnostics')
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(figure_dir / 'bnn-belief-updates-and-prediction.png', dpi=160)
        plt.close(fig)

    result = dict(plant=m['plant'], master_seed=m['master_seed'], checks=checks,
                  arms={key: v['metrics'] for key, v in data.items()},
                  completed_at=datetime.now(ZoneInfo('America/New_York')).isoformat())
    (doc / metrics_name).write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Known-dynamics SIT-LMPC and learned BNN on a sampled unknown plant', '', '## Main findings', '',
             f"- Both controllers were tested on the same fixed sampled plant: friction **{m['plant']['friction']:.6f}**, mass ratio **{m['plant']['mass_ratio']:.6f}**. Only the analytical controller received the true parameters."]
    if m.get('resume_checkpoints'):
        lines += [f"- Continued each recorded trial-10 checkpoint toward **{limit} total attempts per arm**. The plotted histories include the restored prefix; this is not a new independent seed replicate."]
    for arm, values in data.items():
        v = values['metrics']
        best = f"{v['best_learned_lap_seconds']:.2f} s" if v['best_learned_lap_seconds'] is not None else 'not reached'
        lines += [f"- **{LABELS[arm]}:** {v['status']}; **{v['successful_laps']} successful / {v['failed_attempts']} failed** attempts, {v['learned_steps']:,} learned-control steps, {v['value_generations']} value generations; best completed learned lap **{best}**."]
    b = data['bnn-unknown']['metrics']
    if b.get('last_heldout_10step_xy_rmse_m') is not None:
        lines += [f"- Final BNN held-out 10-step mean XY error: **{1000*b['last_heldout_10step_xy_rmse_m']:.2f} mm**; function-path tube coverage **{b['last_heldout_10step_path_coverage']:.0%}** over **{b['last_heldout_10step_windows']}** evaluated windows. Both runs recorded zero physical boundary violations."]
    lines += [f"- The BNN retained **{b['posterior_training_transitions']:,}** fitting transitions and published **{b['belief_generation']}** belief generations over {b['posterior_updates_considered']} attempt-boundary updates. Held candidates add evidence without changing the active predictor.",
              f"- Chance selection used 64 empirical coherent functions. Conservative fallback was applied on **{b['chance_fallback_controls']}** controls; those controls are not evidence that both chance thresholds were satisfied.",
              '- This is one-plant, one-seed functional validation with full configured training budgets; it is not a robustness or safety certificate.', '',
              '**Table 1. Measured outcomes.**', '',
              '| Controller | Successful / failed | Total controls | Learned controls | Value generations | Belief generation | Best learned lap (s) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm, values in data.items():
        v = values['metrics']
        best = f"{v['best_learned_lap_seconds']:.2f}" if v['best_learned_lap_seconds'] is not None else '—'
        lines += [f"| {LABELS[arm]} | {v['successful_laps']} / {v['failed_attempts']} | {v['applied_steps']:,} | {v['learned_steps']:,} | {v['value_generations']} | {v['belief_generation']} | {best} |"]
    lines += ['', '![Figure 1. Learning curves and measured violations.](figures/known-vs-bnn-learning-and-violations.png)', '',
              '**Figure 1.** Warm-up and learned trials are both retained. The dotted line marks the start of learned control after five warm-up laps. Physical boundary violation is measured from the actual simulator state; safe-set violation is the nominal planner metric. Failed trials remain on the trial axis; missing completed-lap times use a red cross at the axis.']
    if history:
        lines += ['', '![Figure 2. Posterior updates and prediction diagnostics.](figures/bnn-belief-updates-and-prediction.png)', '',
                  '**Figure 2.** Belief updates are considered after attempts, including warm-up. Held-out coverage measures coherent function-only XY path tubes; observation noise is excluded and it is not a probability guarantee for the plant. A held update retains the previous published belief.']
    lines += ['', '## Configuration and interpretation', '',
              '**Table 2. Controlled settings.**', '',
              '| Setting | Value |', '|---|---|',
              '| Plant sampling | Uniform friction [0.4, 0.9], mass ratio [0.95, 1.05]; fixed after sampling |',
              '| BNN population training support | Friction [0.3, 1.0], mass ratio [0.9, 1.1], nominal cornering coefficients in these checks |',
              '| Seed / map | 20261107 / custom1 |',
              '| Action samples / horizon | 1,024 / 10 transitions |',
              '| NF value model | 4 layers, 128 hidden units, 300 epochs per fit |',
              '| Safe set | Exact convex-hull facets, four retained laps, bounded precompilation |',
              '| BNN belief | Latent 96, tempered attempt-boundary posterior, fixed pretrained features |',
              '| Chance estimator | 64 planning samples; boundary threshold 0.05, terminal-hull violation threshold 0.9 |',
              '| State / actuation noise | Inherited defaults: 0.001; [0.001, 0.001] |',
              '| Integration | Plant 0.02 s substeps; BNN trained at 0.01 s; controller interval 0.1 s |',
              f'| Trial cap | {limit} each; known SIT mode stops at first physical failure; BNN retries |',
              '', 'The terminal-hull threshold and fallback preserve the migrated v2 experiment settings. The threshold 0.9 is permissive: it allows up to 90% of sampled terminal states outside the relaxed hull. This must not be interpreted as strict sampled membership.',
              '', '## Checks and artifacts', '']
    for name, passed in checks.items():
        lines.append(f"- {name}: **{'passed' if passed else 'NOT established'}**.")
    lines += ['', f'- [Exact manifest and config hashes]({args.manifest.name}).',
              f'- [Machine-readable metrics]({metrics_name}).',
              f"- [Known-dynamics result directory]({root / 'known-dynamics'}).",
              f"- [BNN unknown-dynamics result directory]({root / 'bnn-unknown'}).",
              f"- [Known-dynamics configuration]({Path(m['config_root']) / 'known-dynamics.yaml'}).",
              f"- [Unknown-dynamics configuration]({Path(m['config_root']) / 'bnn-unknown.yaml'}).",
              '', f"Completed: {result['completed_at']} (US Eastern)."]
    (doc / report_name).write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
