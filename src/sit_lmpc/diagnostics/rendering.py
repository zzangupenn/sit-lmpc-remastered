import numpy as np
import jax.numpy as jnp


def density_reference_points(safe_set, frenet_state):
    """Local successful-state data for display; no hull queries or planner RNG."""
    if not safe_set.cartesian:
        return np.empty((0, 2))
    states, cartesian = safe_set.arrays()
    speed = max(float(frenet_state[3]), 1.)
    terminal_s = float(frenet_state[0]) + speed * safe_set.config.horizon * safe_set.timestep
    lo, hi = safe_set.window((terminal_s, terminal_s), speed)
    indices = np.flatnonzero((states[:, 0] > lo) & (states[:, 0] < hi))
    limit = safe_set.config.hull_selection_points
    if len(indices) > limit:
        indices = indices[np.linspace(0, len(indices) - 1, limit, dtype=int)]
    return cartesian[indices, :2]


class TrajectoryRenderer:
    def __init__(self, simulator, track, half_width):
        renderer = simulator.env.unwrapped.renderer
        points = track.centerline.waypoints[:: max(1, len(track.centerline.waypoints) // 200)]
        self.line = renderer.get_closed_lines_renderer(
            points[:, 1:3], color=(80, 80, 80), size=2, z_offset=0.01
        )
        self.boundaries = []
        for offset in (-half_width, half_width):
            poses = jnp.column_stack(
                [jnp.asarray(points[:, 0]), jnp.full(len(points), offset), jnp.zeros(len(points))]
            )
            boundary = np.asarray(track.vmap_frenet_to_cartesian_jax(poses))
            self.boundaries.append(
                renderer.get_closed_lines_renderer(
                    boundary[:, :2], color=(255, 120, 0), size=2, z_offset=0.01
                )
            )
        self.predicted = renderer.get_points_renderer(
            np.zeros((1, 2)), color=(255, 230, 0), size=10, z_offset=0.02
        )
        self.reference = renderer.get_points_renderer(
            np.zeros((1, 2)), color=(255, 0, 180), size=8, z_offset=0.02
        )

    def update(self, result, *, safe_set=None, frenet_state=None):
        reference = result.reference
        if (result.metrics.get("safe_set_query_active") == "nf_density"
                and safe_set is not None and frenet_state is not None):
            reference = density_reference_points(safe_set, frenet_state)
        self.display(dict(trajectory=np.asarray(result.trajectory)[:, :2],
                          reference=np.empty((0, 2)) if reference is None else reference[:, :2]))

    def display(self, data):
        self._display = {key: np.asarray(value).copy() for key, value in data.items()}
        self.predicted.update(self._display["trajectory"])
        self.reference.update(self._display["reference"])

    def snapshot(self):
        return {key: value.copy() for key, value in self._display.items()}
