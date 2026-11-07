import subprocess
import sys
from pathlib import Path


def test_core_does_not_import_optional_gp_or_reference_package():
    script = """
import sit_lmpc
from sit_lmpc.experiments.runner import run
import sys
assert "gpjax" not in sys.modules
assert "numpyro" not in sys.modules
assert "twit_lmpc" not in sys.modules
assert "matplotlib.pyplot" not in sys.modules
assert "utilsuite" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_removed_implementations_are_not_present():
    import sit_lmpc

    root = Path(sit_lmpc.__file__).parent
    for path in root.rglob("*.py"):
        if "/models/gp/" in str(path):
            continue
        text = path.read_text()
        assert "class BayesianPNN" not in text
        assert "class EnsembleBNN" not in text
        assert "class KernelResidualData" not in text
        assert "class SparseGPResidualData" not in text
        assert "import gpjax" not in text
