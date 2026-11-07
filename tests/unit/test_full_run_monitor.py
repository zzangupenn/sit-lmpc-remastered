"""Resumed event histories must not count the checkpoint prefix twice."""
import importlib.util
import json
from pathlib import Path


def test_resume_prefix_is_counted_once_in_live_status(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / 'scripts/monitor_known_bnn_full.py'
    spec = importlib.util.spec_from_file_location('full_monitor', script)
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    monkeypatch.setattr(monitor, 'unit_state', lambda name: {'ActiveState': 'active'})
    doc, output, parent = (tmp_path / name for name in ('docs', 'output', 'parent'))
    for p in (doc, output, parent):
        p.mkdir()
    checkpoints = {}
    for arm in ('known-dynamics', 'bnn-unknown'):
        checkpoint = tmp_path / f'checkpoint-{arm}'
        checkpoint.mkdir()
        (checkpoint/'state.json').write_text(json.dumps(dict(
            trials=10, counts={'success': 10, 'failure': 0}, value_generation=9, belief_generation=None)))
        checkpoints[arm] = {'checkpoint': str(checkpoint)}
        (parent/arm).mkdir()
        (parent/arm/'lap_metrics.json').write_text(json.dumps([
            dict(attempt_id=i, completed=True, lap_time=40.) for i in range(1, 11)]))
        (output/arm).mkdir()
        rows = []
        for i in range(1, 12):
            rows += [dict(event='attempt', attempt_id=i, outcome='success'),
                     dict(event='lap', lap=i, laptime=40.)]
        (output/arm/'events.jsonl').write_text('\n'.join(json.dumps(v) for v in rows))
    manifest = dict(output_root=str(output), documentation_root=str(doc), parent_check_root=str(parent),
        resume_checkpoints=checkpoints, units={v: v for v in checkpoints}, plant={'friction': .8, 'mass_ratio': 1.},
        report_filename='final.md')
    result = monitor.write_status(manifest)
    assert all(v['completed_trials'] == 11 and v['successful_laps'] == 11 for v in result.values())
    text = (doc/'full-known-versus-bnn-live-status.md').read_text()
    assert '22/600 total attempts recorded' in text
