"""Runtime settings are applied by entry points, never by package imports."""

import os
from pathlib import Path


def configure(config, cpu=False):
    cache = Path(".cache").resolve()
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    import jax

    jax.config.update("jax_enable_x64", config.simulation.dtype == "float64")
    jax.config.update("jax_compilation_cache_dir", str(cache / "jax"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.1)
