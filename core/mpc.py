"""Model Predictive Control (MPC) of the motor speed.

Prediction model (discrete, zero-order hold, sample time Ts), states x = [ω, iₐ], input u = Vₐ:
    x_{k+1} = A_d x_k + B_d u_k,      y_k = C x_k = ω_k

The decision variables are the input changes Δu = u_k − u_{k−1}, so the model is augmented with the
previous input:  z_k = [ω_k, iₐ_k, u_{k−1}]ᵀ
    z_{k+1} = Ã z_k + B̃ Δu_k,   Ã = [[A_d, B_d], [0, 1]],   B̃ = [B_d; 1],   C̃ = [1, 0, 0]

Stacking the N-step prediction:
    Ŷ = Φ z_k + Γ ΔU,     Û = 1·u_{k−1} + S ΔU
    Φ (row j) = C̃ Ã^j,   Γ[j, i] = C̃ Ã^(j−i) B̃ (i ≤ j),   S = lower-triangular matrix of ones

Optimisation (a quadratic program) solved every sample:
    min  q‖Ŷ − ω*·1‖² + r‖ΔU‖² + ρ ε²
    s.t. −V_max ≤ Û ≤ V_max
         ω_min − ε ≤ Ŷ ≤ ω_max + ε,   ε ≥ 0
The speed limits are "soft" (slack ε with a very large weight ρ): they hold whenever they can be met,
and the problem never becomes infeasible. Only the first input change is applied (receding horizon).
"""

import math
import time

import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import expm

from .model import motor, state_space


def mpc_params(cfg):
    return dict(Ts=float(cfg["mpc_Ts"]), N=int(cfg["mpc_N"]), q=float(cfg["mpc_q"]), r=float(cfg["mpc_r"]),
                Vmax=float(cfg["mpc_Vmax"]), wmin=float(cfg["mpc_wmin"]), wmax=float(cfg["mpc_wmax"]))


def discrete_model(cfg, Ts):
    """Two-state speed model x = [ω, iₐ], discretised exactly (zero-order hold)."""
    A3, B3, _ = state_space(motor(cfg))
    Ac, Bc = A3[1:, 1:], B3[1:]
    M = np.zeros((3, 3))
    M[:2, :2], M[:2, 2] = Ac, Bc
    Md = expm(M * Ts)
    return Ac, Bc, Md[:2, :2], Md[:2, 2]


def prediction_matrices(Ad, Bd, N):
    At = np.zeros((3, 3))
    At[:2, :2], At[:2, 2], At[2, 2] = Ad, Bd, 1.0
    Bt = np.array([Bd[0], Bd[1], 1.0])
    Ct = np.array([1.0, 0.0, 0.0])
    Phi = np.zeros((N, 3))
    h = np.zeros(N)                      # impulse response C̃ Ã^m B̃, m = 0..N−1
    Ap = np.eye(3)
    for j in range(N):
        h[j] = Ct @ Ap @ Bt
        Ap = Ap @ At
        Phi[j] = Ct @ Ap                 # C̃ Ã^(j+1)
    Gam = np.zeros((N, N))
    for j in range(N):
        Gam[j, :j + 1] = h[j::-1]
    S = np.tril(np.ones((N, N)))
    return At, Bt, Ct, Phi, Gam, S


def unconstrained_law(At, Bt, Phi, Gam, q, r):
    """Without constraints the optimum is linear: Δu_k = −K z_k + K_r ω*."""
    N = Gam.shape[0]
    M = np.linalg.solve(q * Gam.T @ Gam + r * np.eye(N), q * Gam.T)
    K = M[0] @ Phi
    Kr = M[0] @ np.ones(N)
    poles = np.linalg.eigvals(At - np.outer(Bt, K))
    return K, Kr, poles


class MPC:
    def __init__(self, cfg):
        p = mpc_params(cfg)
        self.p = p
        self.Ac, self.Bc, self.Ad, self.Bd = discrete_model(cfg, p["Ts"])
        self.At, self.Bt, self.Ct, self.Phi, self.Gam, self.S = prediction_matrices(self.Ad, self.Bd, p["N"])
        self.K, self.Kr, self.poles = unconstrained_law(self.At, self.Bt, self.Phi, self.Gam, p["q"], p["r"])
        N, q, r = p["N"], p["q"], p["r"]
        self.rho = 1e3 * max(q, 1e-6)                 # slack weight: speed limits almost hard
        H = 2 * (q * self.Gam.T @ self.Gam + r * np.eye(N))
        P = np.zeros((N + 1, N + 1))
        P[:N, :N], P[N, N] = H, 2 * self.rho
        A = np.zeros((3 * N + 1, N + 1))
        A[:N, :N] = self.S                            # voltage limits on Û
        A[N:2 * N, :N], A[N:2 * N, N] = self.Gam, -1  # Ŷ − ε ≤ ω_max
        A[2 * N:3 * N, :N], A[2 * N:3 * N, N] = self.Gam, 1  # Ŷ + ε ≥ ω_min
        A[3 * N, N] = 1                               # ε ≥ 0
        self.H, self.Acon = H, A
        self.solver = osqp.OSQP()
        self.solver.setup(P=sparse.triu(sparse.csc_matrix(P)).tocsc(), q=np.zeros(N + 1),
                          A=sparse.csc_matrix(A), l=-np.inf * np.ones(3 * N + 1), u=np.inf * np.ones(3 * N + 1),
                          verbose=False, eps_abs=1e-6, eps_rel=1e-6, polishing=True, max_iter=20000, check_termination=10,
                          warm_starting=True)

    def solve(self, x, u_prev, ref):
        """x = [ω, iₐ] measured now, u_prev = input applied at the previous step. Returns (Δu sequence, slack)."""
        p, N = self.p, self.p["N"]
        z = np.array([x[0], x[1], u_prev])
        free = self.Phi @ z                          # predicted speed if Δu = 0
        f = np.zeros(N + 1)
        f[:N] = 2 * p["q"] * self.Gam.T @ (free - ref)
        f[N] = 10 * self.rho
        inf = np.inf * np.ones(N)
        lo = np.concatenate([-p["Vmax"] - u_prev * np.ones(N), -inf, p["wmin"] - free, [0.0]])
        hi = np.concatenate([p["Vmax"] - u_prev * np.ones(N), p["wmax"] - free, inf, [np.inf]])
        self.solver.update(q=f, l=lo, u=hi)
        res = self.solver.solve()
        sol = res.x
        if sol is None or not np.all(np.isfinite(sol)):      # should not happen (problem is always feasible)
            return np.zeros(N), np.nan
        return sol[:N], float(sol[N])


def simulate_mpc(cfg):
    """Closed loop: MPC samples every Ts; the motor (3 states θ, ω, iₐ) is advanced exactly in between."""
    ctrl = MPC(cfg)
    p = ctrl.p
    Ts, T = p["Ts"], float(cfg["t_end"])
    m = max(1, int(round(Ts / min(Ts, T / 4000))))   # plant sub-steps per controller sample
    dt = Ts / m
    A3, B3, _ = state_space(motor(cfg))
    M = np.zeros((4, 4))
    M[:3, :3], M[:3, 3] = A3, B3
    Md = expm(M * dt)
    Ad3, Bd3 = Md[:3, :3], Md[:3, 3]
    n_samples = int(math.ceil(T / Ts))
    n = n_samples * m
    t = np.arange(n + 1) * dt
    X = np.zeros((n + 1, 3))
    V = np.zeros(n + 1)
    x = np.zeros(3)
    u_prev, ref = 0.0, float(cfg["ref"])
    slack_max, t0 = 0.0, time.perf_counter()
    plans = []
    for s in range(n_samples):
        dU, eps = ctrl.solve(x[1:], u_prev, ref)
        u = float(np.clip(u_prev + dU[0], -p["Vmax"], p["Vmax"]))
        slack_max = max(slack_max, eps if np.isfinite(eps) else 0.0)
        if s in (0, n_samples // 10):
            plans.append(dict(t=s * Ts, U=u_prev + np.cumsum(dU), Y=ctrl.Phi @ np.array([x[1], x[2], u_prev]) + ctrl.Gam @ dU))
        for i in range(m):
            k = s * m + i
            X[k], V[k] = x, u
            x = Ad3 @ x + Bd3 * u
        u_prev = u
    X[n], V[n] = x, u_prev
    return dict(kind="MPC", t=t, theta=X[:, 0], omega=X[:, 1], ia=X[:, 2], va=V, ref=ref, blew_up=False,
                solve_ms=1000 * (time.perf_counter() - t0) / n_samples, slack_max=slack_max, plans=plans,
                Ts=Ts, N=p["N"])


def mpc_summary(cfg):
    """Numbers shown in the app and report: discrete model, unconstrained gain, its poles."""
    c = MPC(cfg)
    stable = bool(np.all(np.abs(c.poles) < 1 - 1e-9))
    return dict(Ac=c.Ac, Bc=c.Bc, Ad=c.Ad, Bd=c.Bd, K=c.K, Kr=c.Kr, poles=c.poles, stable=stable, p=c.p)
