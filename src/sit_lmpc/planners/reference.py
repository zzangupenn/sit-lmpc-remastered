"""Original reference interpolation, retained for numerical compatibility."""

import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def get_reference_trajectory_jax(
    predicted_speeds, dist_from_segment_start, idx, waypoints, n_steps, waypoints_distances, DT
):
    total_length = jnp.sum(waypoints_distances)
    s_relative = jnp.concatenate(
        [jnp.array([dist_from_segment_start]), predicted_speeds * DT]
    ).cumsum()
    s_relative = s_relative % total_length
    rolled_distances = jnp.roll(waypoints_distances, -idx)
    wp_dist_cum = jnp.concatenate([jnp.zeros(1, dtype=waypoints.dtype), jnp.cumsum(rolled_distances)])
    index_relative = jnp.searchsorted(wp_dist_cum, s_relative, side="right") - 1
    index_relative = jnp.clip(index_relative, 0, len(rolled_distances) - 1)
    index_absolute = (idx + index_relative) % (waypoints.shape[0] - 1)
    next_index = (index_absolute + 1) % (waypoints.shape[0] - 1)
    seg_start = wp_dist_cum[index_relative]
    seg_len = rolled_distances[index_relative]
    t = (s_relative - seg_start) / seg_len
    p0 = waypoints[index_absolute][:, 1:3]
    p1 = waypoints[next_index][:, 1:3]
    interpolated_positions = p0 + (p1 - p0) * t[:, jnp.newaxis]
    s0 = waypoints[index_absolute][:, 0]
    s1 = waypoints[next_index][:, 0]
    interpolated_s = (s0 + (s1 - s0) * t) % waypoints[-1, 0]
    yaw0 = waypoints[index_absolute][:, 3]
    yaw1 = waypoints[next_index][:, 3]
    interpolated_yaw = yaw0 + (yaw1 - yaw0) * t
    interpolated_yaw = (interpolated_yaw + jnp.pi) % (2 * jnp.pi) - jnp.pi
    v0 = waypoints[index_absolute][:, 5]
    v1 = waypoints[next_index][:, 5]
    interpolated_speed = v0 + (v1 - v0) * t
    reference = jnp.stack(
        [
            interpolated_positions[:, 0],
            interpolated_positions[:, 1],
            interpolated_speed,
            interpolated_yaw,
            interpolated_s,
            jnp.zeros_like(interpolated_speed),
            jnp.zeros_like(interpolated_speed),
        ],
        axis=1,
    )
    return reference


@jax.jit
def nearest_point_jax(point, trajectory, diffs):
    # diffs = trajectory[1:] - trajectory[:-1]
    l2s = jnp.sum(diffs**2, axis=1) + 1e-8
    dots = jnp.sum((point - trajectory[:-1]) * diffs, axis=1)
    t = jnp.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1] + diffs * t[:, None]
    dists = jnp.linalg.norm(point - projections, axis=1)
    min_dist_segment = jnp.argmin(dists)
    dist_from_segment_start = jnp.linalg.norm(diffs[min_dist_segment] * t[min_dist_segment])
    return (
        projections[min_dist_segment],
        dist_from_segment_start,
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )
