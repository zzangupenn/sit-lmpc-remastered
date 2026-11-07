"""F1TENTH Gym profile constants; see NOTICE for upstream attribution."""

STATE_FIELDS = ("x", "y", "steering_angle", "speed", "yaw", "yaw_rate", "slip_angle")
CONTROL_FIELDS = ("steering_speed", "acceleration")
PARAMETER_FIELDS = (
    "mu",
    "C_Sf",
    "C_Sr",
    "lf",
    "lr",
    "h",
    "m",
    "I",
    "s_min",
    "s_max",
    "sv_min",
    "sv_max",
    "v_switch",
    "a_max",
    "v_min",
    "v_max",
    "width",
    "length",
)


def vehicle_parameters(profile="fullscale", friction=None, mass_ratio=1.0, inertia_policy="fixed"):
    if profile == "fullscale":
        values = (
            1.0489,
            20.89,
            20.89,
            0.88392,
            1.50876,
            0.557,
            1225.8878467253344,
            1538.8533713561394,
            -0.91,
            0.91,
            -0.4,
            0.4,
            4.755,
            3.5,
            -13.9,
            45.8,
            1.674,
            4.298,
        )
    elif profile == "f1tenth":
        values = (
            1.0489,
            4.718,
            5.4562,
            0.15875,
            0.17145,
            0.074,
            3.74,
            0.04712,
            -0.4189,
            0.4189,
            -3.2,
            3.2,
            7.319,
            9.51,
            -5.0,
            20.0,
            0.31,
            0.58,
        )
    else:
        raise ValueError(f"Unknown vehicle profile: {profile}")
    if mass_ratio <= 0 or (friction is not None and friction <= 0):
        raise ValueError("friction and mass ratio must be positive")
    if inertia_policy not in {"fixed", "proportional"}:
        raise ValueError("invalid inertia_policy")
    parameters = dict(zip(PARAMETER_FIELDS, values, strict=True))
    parameters["m"] *= mass_ratio
    if inertia_policy == "proportional":
        parameters["I"] *= mass_ratio
    if friction is not None:
        parameters["mu"] = friction
    return parameters
