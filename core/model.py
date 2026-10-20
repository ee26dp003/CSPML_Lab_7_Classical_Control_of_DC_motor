"""DC motor model and closed-loop maths.

    Va = R·ia + L·dia/dt + Kb·ω        (armature circuit)
    J·dω/dt + B·ω = Ka·ia              (rotor)
    dθ/dt = ω

    G(s) = Ω(s)/Va(s) = Ka / [ JL s² + (BL+RJ) s + (RB+KaKb) ]
"""

import numpy as np

from .config import gains_of


def motor(cfg):
    return {k: float(cfg[k]) for k in ("R", "L", "J", "B", "Ka", "Kb")}


def plant_tf(p):
    num = [p["Ka"]]
    den = [p["J"] * p["L"], p["B"] * p["L"] + p["R"] * p["J"], p["R"] * p["B"] + p["Ka"] * p["Kb"]]
    return num, den


def state_space(p):
    """x = [θ, ω, ia],  ẋ = A x + B Va,  y = ω."""
    R, L, J, B, Ka, Kb = (p[k] for k in ("R", "L", "J", "B", "Ka", "Kb"))
    A = np.array([[0, 1, 0], [0, -B / J, Ka / J], [0, -Kb / L, -R / L]], float)
    Bv = np.array([0, 0, 1 / L], float)
    C = np.array([0, 1, 0], float)
    return A, Bv, C


def motor_summary(p):
    num, den = plant_tf(p)
    a1 = den[2]
    return dict(num=num, den=den, poles=np.roots(den), dc_gain=p["Ka"] / a1,
                tau_e=p["L"] / p["R"], tau_m=p["R"] * p["J"] / a1)


def closed_loop(cfg, kind):
    """Closed-loop transfer function Y/R = C·G/(1 + C·G), its poles and stability (Routh)."""
    p = motor(cfg)
    g = gains_of(cfg, kind)
    Kp, Ki, Kd = g["Kp"], g["Ki"], g["Kd"]
    JL, a2, a1 = plant_tf(p)[1]
    Ka = p["Ka"]
    if Ki > 0:
        num = [Ka * Kd, Ka * Kp, Ka * Ki]
        den = [JL, a2 + Ka * Kd, a1 + Ka * Kp, Ka * Ki]
    else:
        num = [Ka * Kd, Ka * Kp]
        den = [JL, a2 + Ka * Kd, a1 + Ka * Kp]
    while len(num) > 1 and num[0] == 0:
        num = num[1:]
    poles = np.roots(den)
    # Routh–Hurwitz: 2nd order -> all coefficients > 0; 3rd order -> also a2·a1 > a3·a0
    stable = all(c > 0 for c in den) and (len(den) < 4 or den[1] * den[2] > den[0] * den[3])
    if Ki > 0:
        kp_min = JL * Ki / (a2 + Ka * Kd) - p["R"] * p["B"] / Ka - p["Kb"]
        ess = 0.0
    else:
        kp_min = -a1 / Ka
        ess = float(cfg["ref"]) * a1 / (a1 + Ka * Kp)
    return dict(gains=g, num=num, den=den, poles=poles, stable=stable, kp_min=kp_min, ess_theory=ess)
