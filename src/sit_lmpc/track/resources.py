from pathlib import Path
from importlib.resources import files


def map_path(name_or_path):
    name = str(name_or_path)
    if name in {"custom1", "Spielberg"}:
        return Path(str(files("sit_lmpc.track") / "maps" / name / name))
    path = Path(name).expanduser().resolve()
    if not path.with_name(path.name + "_map.yaml").exists():
        raise FileNotFoundError(
            f"Map prefix not found: {path}; use custom1, Spielberg, or a local map prefix"
        )
    return path
