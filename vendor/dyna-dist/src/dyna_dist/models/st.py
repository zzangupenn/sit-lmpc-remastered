"""Gym-compatible single-track RK4 in JAX; upstream attribution is in NOTICE.

This is the nominal residual baseline, never the source of collection labels.
Unlike older BRO equations, the low-speed branch matches this Gym fork exactly.
"""

from functools import partial
import jax
import jax.numpy as jnp


@jax.jit
def derivative(state, control, parameters):
    mu, csf, csr, lf, lr, height, mass, inertia = parameters[:8]
    s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max = parameters[8:16]
    delta, speed, yaw, rate, beta = state[2:]
    steering = jnp.where(
        ((delta <= s_min) & (control[0] <= 0)) | ((delta >= s_max) & (control[0] >= 0)),
        0,
        jnp.clip(control[0], sv_min, sv_max),
    )
    safe_speed = jnp.maximum(speed, 0.5)
    accel_limit = jnp.where(speed > v_switch, a_max * v_switch / safe_speed, a_max)
    acceleration = jnp.where(
        ((speed <= v_min) & (control[1] <= 0)) | ((speed >= v_max) & (control[1] >= 0)),
        0,
        jnp.clip(control[1], -a_max, accel_limit),
    )
    length = lf + lr

    def kinematic(_):
        beta_hat = jnp.arctan(jnp.tan(delta) * lr / length)
        beta_dot = (
            lr
            * steering
            / (length * jnp.cos(delta) ** 2 * (1 + (jnp.tan(delta) * lr / length) ** 2))
        )
        yaw_acceleration = (
            acceleration * jnp.cos(beta) * jnp.tan(delta)
            - speed * jnp.sin(beta) * jnp.tan(delta) * beta_dot
            + speed * jnp.cos(beta) * steering / jnp.cos(delta) ** 2
        ) / length
        return jnp.array(
            [
                speed * jnp.cos(yaw + beta_hat),
                speed * jnp.sin(yaw + beta_hat),
                steering,
                acceleration,
                speed * jnp.cos(beta_hat) * jnp.tan(delta) / length,
                yaw_acceleration,
                beta_dot,
            ]
        )

    def dynamic(_):
        front = 9.81 * lr - acceleration * height
        rear = 9.81 * lf + acceleration * height
        yaw_acceleration = (
            mu
            * mass
            / (inertia * length)
            * (
                lf * csf * front * delta
                + (lr * csr * rear - lf * csf * front) * beta
                - (lf**2 * csf * front + lr**2 * csr * rear) * rate / safe_speed
            )
        )
        beta_dot = (
            mu
            / (safe_speed * length)
            * (
                csf * front * delta
                - (csr * rear + csf * front) * beta
                + (csr * rear * lr - csf * front * lf) * rate / safe_speed
            )
            - rate
        )
        return jnp.array(
            [
                speed * jnp.cos(yaw + beta),
                speed * jnp.sin(yaw + beta),
                steering,
                acceleration,
                rate,
                yaw_acceleration,
                beta_dot,
            ]
        )

    return jax.lax.cond(speed < 0.5, kinematic, dynamic, None)


@partial(jax.jit, static_argnames=("timestep", "integrator_timestep"))
def step(state, control, parameters, timestep=0.1, integrator_timestep=0.02):
    dt = integrator_timestep

    def substep(_, x):
        k1 = derivative(x, control, parameters)
        k2 = derivative(x + dt * k1 / 2, control, parameters)
        k3 = derivative(x + dt * k2 / 2, control, parameters)
        k4 = derivative(x + dt * k3, control, parameters)
        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    result = jax.lax.fori_loop(0, round(timestep / dt), substep, state)
    return result.at[4].set((result[4] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
