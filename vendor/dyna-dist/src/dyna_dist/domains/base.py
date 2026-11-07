"""Serializable domain identities; no renderer, simulator, or JAX dependency."""

from dataclasses import asdict, dataclass
from dyna_dist.models.profiles import STATE_FIELDS, CONTROL_FIELDS


@dataclass(frozen=True)
class DomainSpec:
    name: str
    state_fields: tuple[str, ...]
    control_fields: tuple[str, ...]
    state_units: tuple[str, ...]
    control_units: tuple[str, ...]
    input_floor: tuple[float, ...]
    target_floor: tuple[float, ...]
    transform: str

    @property
    def state_dimension(self):
        return len(self.state_fields)

    @property
    def control_dimension(self):
        return len(self.control_fields)

    @property
    def input_dimension(self):
        return len(self.input_floor)

    def metadata(self):
        return asdict(self)


ST = DomainSpec(
    "st",
    tuple(STATE_FIELDS),
    tuple(CONTROL_FIELDS),
    ("m", "m", "rad", "m/s", "rad", "rad/s", "rad"),
    ("rad/s", "m/s^2"),
    (0.02, 1.0, 0.02, 0.01, 0.02, 0.2),
    (0.02, 0.005, 0.002, 0.02, 0.002, 0.01, 0.002),
    "st-body-increments-v1",
)
JOINTS = ("rootx", "rootz", "rooty", "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot")
HALFCHEETAH = DomainSpec(
    "halfcheetah",
    tuple(f"qpos/{name}" for name in JOINTS) + tuple(f"qvel/{name}" for name in JOINTS),
    JOINTS[3:],
    ("m", "m") + ("rad",) * 7 + ("m/s", "m/s") + ("rad/s",) * 7,
    ("MuJoCo ctrl (dimensionless)",) * 6,
    (0.02,) + (0.05,) * 7 + (0.5,) * 9 + (0.05,) * 6,
    (0.005,) * 9 + (0.05,) * 9,
    "halfcheetah-qpos-qvel-increments-v1",
)
WALKER_JOINTS = (
    "rootx", "rootz", "rooty", "thigh_joint", "leg_joint", "foot_joint",
    "thigh_left_joint", "leg_left_joint", "foot_left_joint",
)
WALKER2D = DomainSpec(
    "walker2d",
    tuple(f"qpos/{name}" for name in WALKER_JOINTS)
    + tuple(f"qvel/{name}" for name in WALKER_JOINTS),
    WALKER_JOINTS[3:],
    ("m", "m") + ("rad",) * 7 + ("m/s", "m/s") + ("rad/s",) * 7,
    ("MuJoCo ctrl (dimensionless)",) * 6,
    (0.02,) + (0.05,) * 7 + (0.5,) * 9 + (0.05,) * 6,
    (0.002,) * 9 + (0.02,) * 9,
    "walker2d-qpos-qvel-increments-v1",
)
DM_JOINTS = ('rootx', 'world_torso_height', 'rooty', 'right_hip', 'right_knee',
             'right_ankle', 'left_hip', 'left_knee', 'left_ankle')
DM_WALKER = DomainSpec(
    'dm_walker', tuple(f'position/{n}' for n in DM_JOINTS)
    + tuple(f'velocity/{n}' for n in DM_JOINTS), DM_JOINTS[3:],
    WALKER2D.state_units, WALKER2D.control_units, WALKER2D.input_floor,
    WALKER2D.target_floor, 'dm-walker-world-height-x-first-increments-v1',
)


def get_domain(name="st"):
    if name == "st":
        return ST
    if name == "halfcheetah":
        return HALFCHEETAH
    if name == "walker2d":
        return WALKER2D
    if name == 'dm_walker':
        return DM_WALKER
    raise ValueError(f"unsupported dynamics domain: {name}")
