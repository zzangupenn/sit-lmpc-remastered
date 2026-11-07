"""Deterministic regression of the existing normalized lap-time value target."""

import flax.linen as nn
import jax
import jax.numpy as jnp


class MLPValue(nn.Module):
    hidden_dims: int = 128
    layers: int = 4
    activation: str = 'swish'

    @nn.compact
    def __call__(self, inputs):
        hidden = inputs
        for _ in range(self.layers):
            hidden = nn.Dense(self.hidden_dims)(hidden)
            hidden = jnp.sin(hidden) if self.activation == 'sin' else getattr(jax.nn, self.activation)(hidden)
        return nn.Dense(1)(hidden)
