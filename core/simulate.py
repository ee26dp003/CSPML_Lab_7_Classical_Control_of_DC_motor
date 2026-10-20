"""Closed-loop step-response simulation.

The motor equations are advanced with their exact discrete-time form (matrix exponential,
voltage held constant over each very short step), so there is no integration error.
The controller computes the voltage every step from the speed error e = ω* − ω:

    Va = Kp·e + Ki·∫e dt + D,   D = filtered derivative Kd·N·s/(s+N) of e
"""

import math

import numpy as np
from scipy.linalg import expm

from .config import N_FILTER, gains_of
from .model import closed_loop, motor, state_space


def simulate(cfg, kind):
    p = motor(cfg)
    g = gains_of(cfg, kind)
    Kp, Ki, Kd = g["Kp"], g["Ki"], g["Kd"]
    r, T, N = float(cfg["ref"]), float(cfg["t_end"]), N_FILTER

    # step size: much shorter than the fastest closed-loop or motor time constant
    fastest = max(np.max(np.abs(closed_loop(cfg, kind)["poles"])), p["R"] / p["L"], p["B"] / p["J"],
                  N if Kd > 0 else 1.0)
    dt = max(min(T / 5000, 0.05 / fastest), T / 200_000)
    n = int(math.ceil(T / dt))

    A, Bv, _ = state_space(p)
    M = np.zeros((4, 4))
    M[:3, :3], M[:3, 3] = A, Bv
    Md = expm(M * dt)
    Ad, Bd = Md[:3, :3], Md[:3, 3]

    t = np.arange(n + 1) * dt
    X = np.zeros((n + 1, 3))
    V = np.zeros(n + 1)
    x = np.zeros(3)
    z = 0.0          # integral of the error
    D = 0.0          # filtered derivative
    e_prev = 0.0     # error just before the step
    blew_up = False
    for k in range(n + 1):
        e = r - x[1]
        D = (D + Kd * N * (e - e_prev)) / (1 + N * dt) if Kd > 0 else 0.0
        e_prev = e
        u = Kp * e + Ki * z + D
        X[k], V[k] = x, u
        z += e * dt
        x = Ad @ x + Bd * u
        if not np.isfinite(x[1]) or abs(x[1]) > 1e4 * (abs(r) + 1):
            X[k + 1:], V[k + 1:] = np.nan, np.nan
            blew_up = True
            break
    return dict(kind=kind, t=t, theta=X[:, 0], omega=X[:, 1], ia=X[:, 2], va=V, ref=r, blew_up=blew_up)
