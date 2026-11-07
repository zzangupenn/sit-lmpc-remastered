from datetime import datetime
import re
import pytest
import yaml

from sit_lmpc.config import load_config
from sit_lmpc.experiments import runner


class FixedClock:
    @staticmethod
    def now(tz):
        assert tz.key == "America/New_York"
        return datetime(2026, 10, 12, 14, 15, 16, tzinfo=tz)


@pytest.mark.parametrize(
    "map_name,group",
    [("custom1", "custom1"), ("Spielberg", "Spielberg"), ("/maps/my-track/circuit", "circuit")],
)
def test_auto_folder_groups_by_map_and_uses_only_exp_name_and_timestamp(
    tmp_path, monkeypatch, map_name, group
):
    monkeypatch.setattr(runner, "datetime", FixedClock)
    config = load_config(
        overrides={
            "simulation": {"map": map_name, "seed": 987654321},
            "output": {"root": str(tmp_path), "directory": None, "exp_name": "friction-test"},
        }
    )
    directory = runner.output_directory(config)
    assert directory == tmp_path / group / "friction-test-20261012-141516"
    assert "seed" not in directory.name and "987654321" not in directory.name
    saved = yaml.safe_load((directory / "config.yaml").read_text())
    assert saved["simulation"]["seed"] == 987654321
    assert saved["output"]["exp_name"] == "friction-test"
    assert (directory / "metadata.json").is_file()


@pytest.mark.parametrize(
    "utc_instant,expected",
    [
        ("2026-10-13T02:15:16+00:00", "20261012-221516"),
        ("2026-01-13T02:15:16+00:00", "20260112-211516"),
        ("2026-03-08T06:59:59+00:00", "20260308-015959"),
        ("2026-03-08T07:00:00+00:00", "20260308-030000"),
        ("2026-11-01T05:30:00+00:00", "20261101-013000"),
        ("2026-11-01T06:30:00+00:00", "20261101-013000"),
    ],
)
def test_auto_timestamp_uses_eastern_date_and_daylight_saving(
    tmp_path, monkeypatch, utc_instant, expected
):
    class InstantClock:
        @staticmethod
        def now(tz):
            assert tz.key == "America/New_York"
            return datetime.fromisoformat(utc_instant).astimezone(tz)

    monkeypatch.setattr(runner, "datetime", InstantClock)
    config = load_config(
        overrides={"output": {"root": str(tmp_path), "exp_name": "test", "directory": None}}
    )
    assert runner.output_directory(config).name == f"test-{expected}"


def test_repeated_eastern_hour_never_merges_experiments(tmp_path, monkeypatch):
    instants = iter(("2026-11-01T05:30:00+00:00", "2026-11-01T06:30:00+00:00"))

    class InstantClock:
        @staticmethod
        def now(tz):
            return datetime.fromisoformat(next(instants)).astimezone(tz)

    monkeypatch.setattr(runner, "datetime", InstantClock)
    config = load_config(
        overrides={"output": {"root": str(tmp_path), "exp_name": "test", "directory": None}}
    )
    directory = runner.output_directory(config)
    saved_config = (directory / "config.yaml").read_text()
    with pytest.raises(FileExistsError, match="timestamp advances"):
        runner.output_directory(config)
    assert (directory / "config.yaml").read_text() == saved_config


def test_explicit_directory_still_overrides_auto_naming(tmp_path):
    explicit = tmp_path / "exact-manual-location"
    config = load_config(
        overrides={
            "output": {
                "root": str(tmp_path / "unused"),
                "directory": str(explicit),
                "exp_name": "test",
            }
        }
    )
    assert runner.output_directory(config) == explicit
    assert not (tmp_path / "unused").exists()
    marker = explicit / "do-not-overwrite.txt"
    marker.write_text("preserved")
    with pytest.raises(FileExistsError, match="nonempty"):
        runner.output_directory(config)
    assert marker.read_text() == "preserved"


def test_auto_timestamp_collision_never_overwrites_existing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "datetime", FixedClock)
    config = load_config(
        overrides={"output": {"root": str(tmp_path), "directory": None, "exp_name": "test"}}
    )
    path = tmp_path / "custom1" / "test-20261012-141516"
    path.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="timestamp advances"):
        runner.output_directory(config)
    assert not list(path.iterdir())


@pytest.mark.parametrize(
    "name",
    [
        None,
        1,
        "",
        " ",
        ".",
        "..",
        "../other",
        "/absolute",
        "a/b",
        "a\\b",
        "a\nb",
        "a:b",
        " padded ",
    ],
)
def test_invalid_experiment_names_rejected_before_creating_output(tmp_path, name):
    with pytest.raises(ValueError, match="output.exp_name"):
        load_config(overrides={"output": {"root": str(tmp_path), "exp_name": name}})
    assert not list(tmp_path.iterdir())


def test_legacy_yaml_without_exp_name_remains_readable(tmp_path):
    config = tmp_path / "legacy.yaml"
    config.write_text("simulation:\n  seed: 42\noutput:\n  root: results\n")
    loaded = load_config(config)
    assert loaded.output.exp_name == "st"
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", loaded.output.exp_name)
