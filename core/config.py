"""Every tunable value in the lab, with its default, range, unit and a one-line explanation.
The sidebar is built from this list."""

CONTROLLERS = ["P", "PD", "PI", "PID"]

CONTROLLER_NAMES = {
    "P": "Proportional",
    "PD": "Proportional-Derivative",
    "PI": "Proportional-Integral",
    "PID": "Proportional-Integral-Derivative",
}

# Motor parameters (standard textbook DC motor)
MOTOR = [
    # key, label, default, min, max, step, unit, help
    ("R", "Armature resistance R", 1.0, 0.01, 50.0, 0.1, "Ω", "Resistance of the armature winding."),
    ("L", "Armature inductance L", 0.5, 0.0001, 10.0, 0.01, "H",
     "Inductance of the winding. Sets how fast the current can change (electrical time constant L/R)."),
    ("J", "Rotor inertia J", 0.01, 0.00001, 10.0, 0.001, "kg·m²", "Resists changes in speed."),
    ("B", "Viscous friction B", 0.1, 0.0, 10.0, 0.01, "N·m·s/rad", "Friction torque per unit speed."),
    ("Ka", "Torque constant Kₐ", 0.01, 0.0001, 10.0, 0.001, "N·m/A", "Torque produced per ampere: T = Kₐ·iₐ."),
    ("Kb", "Back-EMF constant K_b", 0.01, 0.0001, 10.0, 0.001, "V·s/rad",
     "Voltage the spinning motor generates against the supply: e_b = K_b·ω."),
]

# Selected (design) gains for each controller
DESIGN_GAINS = {
    "P": dict(Kp=100.0),
    "PD": dict(Kp=100.0, Kd=5.0),
    "PI": dict(Kp=50.0, Ki=100.0),
    "PID": dict(Kp=100.0, Ki=200.0, Kd=10.0),
}

GAIN_INFO = {
    "Kp": ("Kp", 0.0, 5000.0, 1.0, "V·s/rad", "Proportional gain: reacts to the present error."),
    "Ki": ("Ki", 0.0, 50000.0, 5.0, "V/rad", "Integral gain: removes the steady-state error."),
    "Kd": ("Kd", 0.0, 500.0, 0.5, "V·s²/rad", "Derivative gain: adds damping, reduces overshoot."),
}

TEST = [
    ("ref", "Reference speed ω*", 1.0, 0.1, 1000.0, 0.1, "rad/s", "Size of the step in target speed."),
    ("t_end", "Simulation time", 2.0, 0.1, 100.0, 0.1, "s", "Long enough for the slowest controller to settle."),
]

# Derivative is built as Kd·N·s/(s+N): a derivative with a small low-pass filter,
# because a pure derivative cannot be implemented.
N_FILTER = 100.0


def default_config():
    cfg = {k: d for k, _, d, *_ in MOTOR}
    cfg.update({k: d for k, _, d, *_ in TEST})
    for c, g in DESIGN_GAINS.items():
        for k, v in g.items():
            cfg[f"{c}_{k}"] = v
    return cfg


def gains_of(cfg, kind):
    return dict(Kp=float(cfg[f"{kind}_Kp"]),
                Ki=float(cfg[f"{kind}_Ki"]) if "I" in kind else 0.0,
                Kd=float(cfg[f"{kind}_Kd"]) if "D" in kind else 0.0)
