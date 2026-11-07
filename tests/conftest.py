import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".cache/numba").resolve()))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import jax

jax.config.update("jax_enable_x64", True)
