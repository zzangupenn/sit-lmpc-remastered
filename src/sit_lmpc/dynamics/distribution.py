"""Coherent trajectory scenarios, independent of the point/ensemble predictor.

Providers freeze one belief generation for a control update. Reusing ``key``
must reuse latent/base draws across every candidate; each scenario retains its
own evolving state. GP/flow backends can implement this contract without changing
the planner. There is deliberately no artificial process/actuator noise here.
"""

from dataclasses import dataclass
from functools import partial
from typing import Protocol

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from .bank import STModelBank, bank_rollout
from .ensemble import STEnsemble


@struct.dataclass
class TrajectoryScenarios:
    """Axes: control sequence, scenario, future step, seven state components.

    ``weights`` has scenario axis only and must already normalize to one; never
    discard invalid mass or multiply posterior density into posterior draws.
    False valid steps denote early-stopped paths, not removed scenarios.
    Exact enumeration has zero probability error. A sampled provider must supply
    a justified simultaneous error envelope valid for adaptively chosen controls
    (and its confidence in metadata); otherwise hard chance verification is
    unavailable. An IID count alone is not such a bound after optimization.
    An explicitly selected empirical planning-sample mode may instead gate raw
    fractions, but must report no confidence/error bound in its diagnostics.
    """

    states: jax.Array
    weights: jax.Array
    valid: jax.Array
    probability_error_bound: jax.Array | None = None


class ScenarioProvider(Protocol):
    scenario_count: int

    @property
    def metadata(self) -> dict: ...

    def rollout(self, state, controls, *, key, scan=True) -> TrajectoryScenarios: ...


@partial(jax.jit, static_argnames=("scan",))
def batch_bank_rollout(bank, state, controls, scan=True):
    return jax.vmap(lambda actions: bank_rollout(bank, state, actions, scan))(controls)


@dataclass(frozen=True)
class STBankScenarioProvider:
    bank: STModelBank
    probabilities: jax.Array
    generation: int = 0
    probability_source: str = "categorical_posterior"

    def __post_init__(self):
        p = np.asarray(self.probabilities)
        if (
            p.shape != (len(self.bank.params),)
            or not np.isfinite(p).all()
            or (p < 0).any()
            or not np.isclose(p.sum(), 1)
        ):
            raise ValueError("scenario probabilities must match the bank and normalize to one")

    @classmethod
    def from_snapshot(cls, snapshot):
        probabilities = snapshot.categorical_weights
        return (
            None
            if probabilities is None
            else cls(
                snapshot.bank,
                probabilities,
                snapshot.generation,
                snapshot.probability_source,
            )
        )

    @property
    def scenario_count(self):
        return len(self.bank.params)

    @property
    def metadata(self):
        return {
            "method": "exact_bank",
            "sample_count": self.scenario_count,
            "probability_source": self.probability_source,
            "belief_generation": int(self.generation),
            "confidence": None,
            "uncertainty": "exact for the declared finite belief; future actuator noise excluded",
        }

    def can_reuse(self, model):
        return isinstance(model, STEnsemble) and model.bank is self.bank

    def from_members(self, members):
        return TrajectoryScenarios(
            members,
            self.probabilities,
            jnp.ones(members.shape[:-1], dtype=bool),
            jnp.asarray(0, dtype=members.dtype),
        )

    def rollout(self, state, controls, *, key=None, scan=True):
        # Enumeration does not use RNG; each friction stays fixed for H steps.
        return self.from_members(batch_bank_rollout(self.bank, state, controls, scan))
