"""Dynamics-distribution learning; collection does not import GPU/plotting backends."""

__version__ = "0.1.0"


def __getattr__(name):
    if name in {"PopulationModel", "Posterior", "FunctionSamples"}:
        from .inference import posterior

        return getattr(posterior, name)
    if name in {"load_model", "load_posterior"}:
        from .io import checkpoint

        return getattr(checkpoint, name)
    raise AttributeError(name)
