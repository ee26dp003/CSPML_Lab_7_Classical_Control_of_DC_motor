"""
DC Motor Speed Control Lab — web front end.        Run with:   streamlit run app.py

All calculations live in core/. This file only collects settings and shows results.
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core import plots  # noqa: E402
from core.config import (CONTROLLER_NAMES, CONTROLLERS, DESIGN_GAINS, GAIN_INFO, MOTOR, MPC, N_FILTER,  # noqa: E402
                         TEST, default_config)
from core.metrics import best_controller, step_metrics  # noqa: E402
from core.model import closed_loop, motor, motor_summary, state_space  # noqa: E402
from core.mpc import mpc_summary, simulate_mpc  # noqa: E402
from core.simulate import simulate  # noqa: E402

ALL = CONTROLLERS + ["MPC"]
CONTROLLER_NAMES = {**CONTROLLER_NAMES, "MPC": "Model Predictive Control"}

st.set_page_config(page_title="DC Motor Speed Control", page_icon="⚙️", layout="wide")

# ------------------------------------------------------------------ settings state
if "cfg_R" not in st.session_state or st.session_state.pop("reset", False):
    for k, v in default_config().items():
        st.session_state["cfg_" + k] = v


def num_input(key, label, lo, hi, step, unit, help_):
    lab = f"{label} [{unit}]" if unit else label
    if isinstance(step, int) and isinstance(lo, int):          # whole numbers (e.g. the horizon N)
        st.number_input(lab, min_value=lo, max_value=hi, step=step, key="cfg_" + key, help=help_)
    else:
        st.number_input(lab, min_value=float(lo), max_value=float(hi), step=float(step),
                        key="cfg_" + key, help=help_, format="%g")


with st.sidebar:
    st.title("⚙️ DC Motor Speed Control")
    st.caption("Change any value; every result updates at once. Hover ⓘ for what a value means.")

    st.subheader("Motor parameters")
    for k, label, _, lo, hi, step, unit, h in MOTOR:
        num_input(k, label, lo, hi, step, unit, h)

    st.subheader("Controller gains")
    for c in CONTROLLERS:
        with st.expander(f"{c} · {CONTROLLER_NAMES[c]}", expanded=False):
            for g in DESIGN_GAINS[c]:
                label, lo, hi, step, unit, h = GAIN_INFO[g]
                num_input(f"{c}_{g}", label, lo, hi, step, unit, h)

    with st.expander("MPC · Model Predictive Control", expanded=False):
        for k, label, _, lo, hi, step, unit, h in MPC:
            num_input(k, label, lo, hi, step, unit, h)

    st.subheader("Test")
    for k, label, _, lo, hi, step, unit, h in TEST:
        num_input(k, label, lo, hi, step, unit, h)

    if st.button("Reset to the selected design values", width="stretch"):
        st.session_state["reset"] = True
        st.rerun()

cfg = {k: st.session_state["cfg_" + k] for k in default_config()}


@st.cache_data(show_spinner="Simulating…", max_entries=50)
def run_all(cfg_items):
    c = dict(cfg_items)
    out = {}
    for k in CONTROLLERS:
        cl = closed_loop(c, k)
        sim = simulate(c, k)
        out[k] = dict(cl=cl, sim=sim, metrics=step_metrics(sim))
    return out


MPC_KEYS = [k for k, *_ in MOTOR] + ["ref", "t_end"] + [k for k, *_ in MPC]


@st.cache_data(show_spinner="Solving the MPC…", max_entries=30)
def run_mpc(cfg_items):
    c = dict(cfg_items)
    sm = mpc_summary(c)
    sim = simulate_mpc(c)
    p_ = sm["p"]
    cl = dict(gains=None, stable=sm["stable"], poles=sm["poles"], ess_theory=0.0, summary=sm)
    label = f"MPC (N={p_['N']}, q={p_['q']:g}, r={p_['r']:g}, ±{p_['Vmax']:g} V)"
    return dict(cl=cl, sim=sim, metrics=step_metrics(sim), label=label)


results = dict(run_all(tuple(sorted((k, v) for k, v in cfg.items() if not k.startswith("mpc_")))))
results["MPC"] = run_mpc(tuple(sorted((k, cfg[k]) for k in MPC_KEYS)))
p = motor(cfg)
M = motor_summary(p)
best = best_controller(results)
best_limited = best_controller(results, only=[k for k in ALL
                                              if np.nanmax(np.abs(results[k]["sim"]["va"])) <= cfg["mpc_Vmax"] + 1e-6])


# ------------------------------------------------------------------ helpers
def poly_tex(c):
    n = len(c) - 1
    terms = []
    for i, a in enumerate(c):
        if a == 0:
            continue
        pw = n - i
        coef = "" if (abs(a) == 1 and pw) else f"{abs(a):.4g}"
        if "e" in coef:
            m_, e_ = coef.split("e")
            coef = f"{m_}\\times10^{{{int(e_)}}}"
        terms.append(("-" if a < 0 else "+", coef + ("" if pw == 0 else ("\\,s" if pw == 1 else f"\\,s^{{{pw}}}"))))
    s = ("-" if terms[0][0] == "-" else "") + terms[0][1]
    return s + "".join(f" {sg} {b}" for sg, b in terms[1:])


def poles_text(ps):
    out, used = [], set()
    ps = sorted(ps, key=lambda z: (-z.real, -z.imag))
    for i, z in enumerate(ps):
        if i in used:
            continue
        if abs(z.imag) > 1e-9:
            used.update(j for j in range(i + 1, len(ps)) if abs(ps[j] - z.conjugate()) < 1e-6 * (1 + abs(z)))
            out.append(f"{z.real:.3g} ± {abs(z.imag):.3g}j")
        else:
            out.append(f"{z.real:.3g}")
    return ", ".join(out)


def gains_text(k, g):
    if k == "MPC":
        return (f"N = {cfg['mpc_N']}, Ts = {cfg['mpc_Ts']:g} s, q = {cfg['mpc_q']:g}, r = {cfg['mpc_r']:g}, "
                f"|Vₐ| ≤ {cfg['mpc_Vmax']:g} V, {cfg['mpc_wmin']:g} ≤ ω ≤ {cfg['mpc_wmax']:g}")
    return ", ".join([f"Kp = {g['Kp']:g}"] + ([f"Ki = {g['Ki']:g}"] if "I" in k else [])
                     + ([f"Kd = {g['Kd']:g}"] if "D" in k else []))


def zero(v, tol=0.05):
    """Treat tiny leftovers (still settling, numerical) as zero."""
    return 0.0 if np.isfinite(v) and abs(v) < tol else v


def fmt(v, d=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


# ------------------------------------------------------------------ header
st.title("Classical speed control of a DC motor")
st.caption("A simulation study of P, PD, PI, PID and Model Predictive Control (MPC) on an armature-controlled DC motor.")

tabs = st.tabs(["📘 Overview", "⚙️ Motor model", "🎛 Controllers", "🔮 MPC", "📈 Step responses", "📊 Comparison"])

# ================================================================== overview
with tabs[0]:
    st.subheader("What this lab does")
    st.markdown("""
We want a DC motor to spin at exactly the speed we ask for, reach it quickly, and not overshoot. The only thing
we can change is the **armature voltage Vₐ**. A **controller** measures the speed, compares it with the target
speed ω\\*, and turns the difference (the **error** e = ω\\* − ω) into a voltage. This is called closed-loop or
feedback control.

We model the motor as a transfer function, try the four classical controllers and a **Model Predictive
Controller (MPC)**, simulate how the speed responds when the target suddenly jumps (a **step input**), and
compare them.
""")
    st.image(os.path.join(HERE, "assets", "loop_block_diagram.png"), width="stretch",
             caption="Closed-loop speed control: the controller C(s) drives the motor G(s); the speed is fed back.")
    st.markdown("""
| Controller | Voltage it applies | What it does |
|---|---|---|
| **P** | Kp·e | Pushes in proportion to the present error. Fast, but always leaves a small steady error. |
| **PD** | Kp·e + Kd·de/dt | Adds a "brake" on fast changes: less overshoot. The steady error stays. |
| **PI** | Kp·e + Ki·∫e dt | Adds memory of past error: removes the steady error, but is slower. |
| **PID** | all three | Fast, accurate and well damped. |
| **MPC** | the first move of an optimised plan | Predicts the next N steps with the motor model and picks the best voltages that respect the voltage and speed limits. |

**How to use this app**
1. Set the motor values, controller gains and MPC settings (including the horizon N) in the sidebar on the left
   (the defaults are the selected design).
2. Open **📈 Step responses** to see the speed, the three state variables and the voltage.
3. Open **📊 Comparison** for the table of rise time, settling time, overshoot and steady-state error, and the
   most suitable controller.
""")
    st.markdown("**Words used**")
    st.markdown("""
| Term | Meaning |
|---|---|
| Rise time | Time to go from 10 % to 90 % of the final speed. |
| Settling time | Time after which the speed stays within ±2 % of its final value. |
| Overshoot | How far the speed goes above its final value, in %. |
| Steady-state error | Target speed minus final speed. |
| Pole | A root of the closed-loop denominator. Negative real part = stable. |
""")

# ================================================================== model
with tabs[1]:
    st.subheader("Motor equations")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Electrical side** (armature circuit)")
        st.latex(r"V_a = R\,i_a + L\,\frac{di_a}{dt} + K_b\,\omega")
    with c2:
        st.markdown("**Mechanical side** (rotor)")
        st.latex(r"J\,\frac{d\omega}{dt} + B\,\omega = K_a\,i_a")
    st.subheader("Transfer function")
    st.markdown("Taking Laplace transforms and eliminating the current gives the speed transfer function:")
    st.latex(r"G(s)=\frac{\Omega(s)}{V_a(s)}=\frac{K_a}{(R+Ls)(Js+B)+K_aK_b}"
             r"=\frac{K_a}{JL\,s^2+(BL+RJ)\,s+(RB+K_aK_b)}")
    st.markdown("With the current motor values:")
    st.latex(rf"G(s)=\frac{{{M['num'][0]:.4g}}}{{{poly_tex(M['den'])}}}")

    st.subheader("State-space model (three state variables)")
    st.markdown("x₁ = θ (position), x₂ = ω (speed), x₃ = iₐ (current); input u = Vₐ; output y = ω.")
    A, Bv, C = state_space(p)
    bm = lambda X: r"\begin{bmatrix}" + r"\\".join("&".join(f"{v:.4g}" for v in row)  # noqa: E731
                                                   for row in np.atleast_2d(X)) + r"\end{bmatrix}"
    st.latex(r"\begin{bmatrix}\dot\theta\\\dot\omega\\\dot i_a\end{bmatrix}=\begin{bmatrix}0&1&0\\0&-\frac{B}{J}&"
             r"\frac{K_a}{J}\\0&-\frac{K_b}{L}&-\frac{R}{L}\end{bmatrix}\begin{bmatrix}\theta\\\omega\\i_a"
             r"\end{bmatrix}+\begin{bmatrix}0\\0\\\frac{1}{L}\end{bmatrix}V_a"
             r"\quad=\quad" + bm(A) + r"x+" + bm(Bv.reshape(-1, 1)) + r"V_a")
    st.image(os.path.join(HERE, "assets", "motor_block_diagram.png"), width="stretch",
             caption="Inside the motor: the back-EMF K_b·ω acts as built-in feedback.")

    st.subheader("Motor characteristics")
    st.dataframe(pd.DataFrame([
        ("Poles of G(s)", poles_text(M["poles"]), "Both negative: the motor on its own is stable."),
        ("DC gain G(0)", f"{M['dc_gain']:.4g} rad/s per V", "Steady speed for 1 V without a controller."),
        ("Electrical time constant L/R", f"{M['tau_e']:.4g} s", "How quickly the current can change."),
        ("Mechanical time constant RJ/(RB+KₐK_b)", f"{M['tau_m']:.4g} s", "How quickly the speed can change."),
    ], columns=["Quantity", "Value", "Meaning"]), hide_index=True, width="stretch")
    if M["tau_e"] > M["tau_m"]:
        st.info(f"Here L/R = {M['tau_e']:.3g} s is slower than the mechanical time constant, so the inductance "
                "matters and the full second-order model is used. Try a smaller L in the sidebar to see the motor "
                "speed up.")

# ================================================================== controllers
with tabs[2]:
    st.subheader("Controller equations")
    st.latex(r"P:\;C(s)=K_p\qquad PD:\;C(s)=K_p+K_d\,s\qquad PI:\;C(s)=K_p+\frac{K_i}{s}"
             r"\qquad PID:\;C(s)=K_p+\frac{K_i}{s}+K_d\,s")
    st.markdown("With unity feedback the closed-loop transfer function is")
    st.latex(r"T(s)=\frac{Y(s)}{R(s)}=\frac{C(s)\,G(s)}{1+C(s)\,G(s)}")
    st.caption("The MPC is formulated separately in the 🔮 MPC tab. "
               f"In the simulation the derivative is built as Kd·N·s/(s+N) with N = {N_FILTER:g} rad/s, a "
               "derivative with a small filter, because a pure derivative cannot be implemented.")

    st.subheader("Selected gains and closed-loop results")
    rows = []
    for k in CONTROLLERS:
        cl = results[k]["cl"]
        rows.append({"Controller": f"{k} · {CONTROLLER_NAMES[k]}", "Gains": gains_text(k, cl["gains"]),
                     "Closed-loop poles": poles_text(cl["poles"]),
                     "Stable (Routh–Hurwitz)": "✅ yes" if cl["stable"] else "❌ no",
                     "Steady error (theory)": "0" if "I" in k and cl["gains"]["Ki"] > 0
                     else f"{cl['ess_theory']:.4g} rad/s"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown("**Closed-loop transfer functions with the current values**")
    for k in CONTROLLERS:
        cl = results[k]["cl"]
        st.latex(rf"T_{{{k}}}(s)=\frac{{{poly_tex(cl['num'])}}}{{{poly_tex(cl['den'])}}}")
        if not cl["stable"]:
            st.error(f"{k} is unstable with these gains. For this Ki, Kp must be above {cl['kp_min']:.3g}.")

    with st.expander("What each gain does"):
        st.markdown("""
- **Kp** (proportional) raises the speed of response and shrinks the steady error, but too much causes overshoot.
- **Ki** (integral) adds a pole at s = 0, which drives the steady-state error to zero. Too much causes overshoot
  and can make the loop unstable.
- **Kd** (derivative) increases the s-term of the denominator, which increases damping and reduces overshoot.
  It does not change the steady-state error.
""")

# ================================================================== MPC
with tabs[3]:
    mp = results["MPC"]
    sm = mp["cl"]["summary"]
    pm = sm["p"]
    st.subheader("What MPC does")
    st.markdown("""
At every sample time the MPC **measures the motor**, **predicts** with the motor model how the speed will evolve
over the next **N** samples, and **chooses the best sequence of voltages** for that window by solving a small
optimisation problem. The best plan keeps the speed close to ω\\* with smooth voltage changes, and never breaks the
voltage and speed limits. Only the **first** voltage of the plan is applied; at the next sample the window moves
forward and the plan is made again (the *receding horizon*).
""")
    st.markdown("**1 · Prediction model.** States x = [ω, iₐ]ᵀ, input u = Vₐ, output y = ω (the position is not "
                "needed for speed control). Continuous model and its exact discretisation with sample time Tₛ:")
    bm2 = lambda X: r"\begin{bmatrix}" + r"\\".join("&".join(f"{v:.5g}" for v in row)  # noqa: E731
                                                     for row in np.atleast_2d(X)) + r"\end{bmatrix}"
    st.latex(r"\dot x=\begin{bmatrix}-\frac{B}{J}&\frac{K_a}{J}\\-\frac{K_b}{L}&-\frac{R}{L}\end{bmatrix}x+"
             r"\begin{bmatrix}0\\\frac{1}{L}\end{bmatrix}u,\quad y=\begin{bmatrix}1&0\end{bmatrix}x"
             r"\quad\Rightarrow\quad x_{k+1}=A_d\,x_k+B_d\,u_k,\quad A_d=e^{AT_s},\;"
             r"B_d=\int_0^{T_s}e^{A\tau}d\tau\,B")
    st.latex(rf"T_s={pm['Ts']:g}\text{{ s}}:\qquad A_d=" + bm2(sm["Ad"]) + r",\qquad B_d=" + bm2(sm["Bd"].reshape(-1, 1)))
    st.markdown("**2 · Optimisation solved at every sample k** (predictions made at time k are written "
                "$\\hat{(\\cdot)}_{k+j|k}$, and $\\Delta\\hat u_{k+j|k}=\\hat u_{k+j|k}-\\hat u_{k+j-1|k}$ with "
                "$\\hat u_{k-1|k}=u_{k-1}$):")
    st.latex(r"\min_{\Delta\hat u_{k|k},\dots,\Delta\hat u_{k+N-1|k}}\;J_k=\sum_{j=1}^{N}q\,"
             r"\big(\hat y_{k+j|k}-\omega^*\big)^2+\sum_{j=0}^{N-1}r\,\big(\Delta\hat u_{k+j|k}\big)^2")
    st.latex(r"\text{s.t.}\quad \hat x_{k|k}=x_k,\quad \hat x_{k+j+1|k}=A_d\hat x_{k+j|k}+B_d\hat u_{k+j|k},"
             r"\quad \hat y_{k+j|k}=C\hat x_{k+j|k}")
    st.latex(r"-V_{max}\le\hat u_{k+j|k}\le V_{max},\qquad \omega_{min}\le\hat y_{k+j+1|k}\le\omega_{max},"
             r"\qquad j=0,\dots,N-1")
    st.markdown("**3 · Apply the first move** $u_k=u_{k-1}+\\Delta\\hat u^*_{k|k}$, measure again at k+1, and repeat.")
    with st.expander("How it is solved: the matrix (quadratic-program) form"):
        st.markdown("With the augmented state $z_k=[\\omega_k,\\;i_{a,k},\\;u_{k-1}]^T$ the whole horizon "
                    "is predicted at once:")
        st.latex(r"\hat Y=\Phi z_k+\Gamma\,\Delta U,\qquad \hat U=\mathbf 1\,u_{k-1}+S\,\Delta U,"
                 r"\qquad \tilde A=\begin{bmatrix}A_d&B_d\\0&1\end{bmatrix},\;\tilde B=\begin{bmatrix}B_d\\1"
                 r"\end{bmatrix},\;\tilde C=\begin{bmatrix}1&0&0\end{bmatrix}")
        st.latex(r"\Phi=\begin{bmatrix}\tilde C\tilde A\\\tilde C\tilde A^2\\\vdots\\\tilde C\tilde A^N"
                 r"\end{bmatrix},\quad \Gamma=\begin{bmatrix}\tilde C\tilde B&0&\cdots&0\\\tilde C\tilde A"
                 r"\tilde B&\tilde C\tilde B&\cdots&0\\\vdots&&\ddots&\\\tilde C\tilde A^{N-1}\tilde B&"
                 r"\cdots&&\tilde C\tilde B\end{bmatrix},\quad S=\begin{bmatrix}1&0&\cdots&0\\1&1&\cdots&0"
                 r"\\\vdots&&\ddots&\\1&1&\cdots&1\end{bmatrix}")
        st.markdown("Substituting into the cost gives a quadratic program in ΔU:")
        st.latex(r"J=\tfrac12\Delta U^TH\,\Delta U+f^T\Delta U,\quad H=2\,(q\,\Gamma^T\Gamma+r\,I),\quad "
                 r"f=2q\,\Gamma^T(\Phi z_k-\omega^*\mathbf 1)")
        st.latex(r"-V_{max}\mathbf 1-u_{k-1}\mathbf 1\le S\,\Delta U\le V_{max}\mathbf 1-u_{k-1}\mathbf 1,"
                 r"\qquad \omega_{min}\mathbf 1-\Phi z_k\le\Gamma\,\Delta U\le\omega_{max}\mathbf 1-\Phi z_k")
        st.markdown("The QP is solved with OSQP every sample. The speed limits are made *soft* with a slack "
                    "variable that has a very large penalty, so the problem can never become infeasible; the slack "
                    f"stayed at {mp['sim']['slack_max']:.1e} rad/s in this run, i.e. the speed limits were met.")
        st.markdown("**Without constraints** the optimum is a fixed linear law $\\Delta u_k=-Kz_k+K_r\\omega^*$ "
                    "(the first row of $H^{-1}$ applied to $-f$). Its closed-loop poles show the MPC is stable when "
                    "no limit is active:")
        st.latex(r"K=" + bm2(sm["K"].reshape(1, -1)) + rf",\qquad K_r={sm['Kr']:.5g},\qquad z\text{{-plane poles: }}"
                 + ", ".join(f"{abs(z):.4f}" + (r"\angle" + f"{np.degrees(np.angle(z)):.1f}" + r"^\circ" if abs(z.imag) > 1e-9 else "")
                             for z in sm["poles"]))
    st.subheader("Current MPC settings")
    st.dataframe(pd.DataFrame([
        ("Prediction horizon N", f"{pm['N']} steps", f"looks {pm['N'] * pm['Ts']:.3g} s ahead"),
        ("Sample time Tₛ", f"{pm['Ts']:g} s", "how often the plan is remade"),
        ("Weights q / r", f"{pm['q']:g} / {pm['r']:g}", "tracking error vs. voltage changes"),
        ("Voltage limit", f"±{pm['Vmax']:g} V", "hard limit"),
        ("Speed limits", f"{pm['wmin']:g} … {pm['wmax']:g} rad/s", "caps undershoot / overshoot"),
        ("Stable without constraints", "✅ yes" if sm["stable"] else "❌ no", "all poles inside the unit circle"),
        ("Average solve time", f"{mp['sim']['solve_ms']:.2f} ms", f"per sample (must be ≪ Tₛ = {pm['Ts'] * 1000:g} ms)"),
    ], columns=["Setting", "Value", "Meaning"]), hide_index=True, width="stretch")
    if pm["wmax"] < cfg["ref"]:
        st.warning(f"ω_max = {pm['wmax']:g} rad/s is below the reference ω* = {cfg['ref']:g} rad/s, so the MPC "
                   "cannot reach the target. Set ω_max a little above ω*.")
    if mp["sim"]["plans"]:
        st.subheader("The receding horizon in action")
        plan = mp["sim"]["plans"][-1]
        st.plotly_chart(plots.mpc_plan(plan, mp["sim"], cfg["ref"], pm["wmax"], pm["Ts"]), width="stretch")
        st.caption(f"Dots: the plan the MPC makes at t = {plan['t']:.2f} s for the next N samples (shaded window). "
                   "Faint thick line: what actually happens. The plan keeps the voltage at the limit while the "
                   "motor accelerates, then lowers it so the speed lands on ω* without crossing the red dotted "
                   "ω_max line. Only the first planned voltage is applied; the plan is remade every Tₛ. Because "
                   "the model matches the motor, the prediction and the real response coincide.")

# ================================================================== responses
with tabs[4]:
    show = st.multiselect("Controllers to show", ALL, default=ALL)
    if not show:
        st.info("Choose at least one controller.")
    else:
        st.plotly_chart(plots.speed(results, show), width="stretch")
        st.caption("The dashed line is the target speed. P and PD level off below it (steady-state error); PI and "
                   "PID end on it. The higher a curve goes above its final value, the larger the overshoot.")
        st.markdown("**The three state variables**")
        st.plotly_chart(plots.states(results, show), width="stretch")
        st.caption("Position θ rises with a slope equal to the speed. The current iₐ peaks while the motor "
                   "accelerates, then settles at the value needed to overcome friction.")
        st.plotly_chart(plots.voltage(results, show, cfg["mpc_Vmax"]), width="stretch")
        st.caption("The voltage each controller applies to the motor. A derivative term gives a large, very short "
                   "spike at the moment the target jumps. The MPC never leaves its ± voltage limit (dotted lines).")

# ================================================================== comparison
with tabs[5]:
    st.subheader("Comparison of performance metrics")
    rows = []
    for k in ALL:
        m, cl = results[k]["metrics"], results[k]["cl"]
        ok = cl["stable"] and not results[k]["sim"]["blew_up"]
        gtxt = (f"N={cfg['mpc_N']}, Ts={cfg['mpc_Ts']:g}, q={cfg['mpc_q']:g}, r={cfg['mpc_r']:g}"
                if k == "MPC" else gains_text(k, cl["gains"]))
        rows.append({"Controller": k, "Gains": gtxt,
                     "Rise time (s)": m["tr"] if ok else np.nan, "Settling time (s)": m["ts"] if ok else np.nan,
                     "Overshoot (%)": m["Mp"] if ok else np.nan, "Final speed (rad/s)": m["y_final"] if ok else np.nan,
                     "Steady-state error (%)": zero(m["ess_pct"]) if ok else np.nan,
                     "Peak voltage (V)": float(np.nanmax(np.abs(results[k]["sim"]["va"]))),
                     "Stable": "yes" if ok else "no"})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in df.columns
                                if c not in ("Controller", "Gains", "Stable")})
    st.plotly_chart(plots.metric_bars(results, ALL), width="stretch")
    st.download_button("⬇ Download table (CSV)", df.to_csv(index=False), file_name="controller_comparison.csv")

    st.subheader("Observations")
    obs = []
    R = {k: results[k]["metrics"] for k in ALL}
    stable = {k: results[k]["cl"]["stable"] and not results[k]["sim"]["blew_up"] for k in ALL}
    for k in ALL:
        if not stable[k]:
            obs.append(f"**{k}** is unstable with these gains, so it cannot be used.")
    if stable["P"]:
        obs.append(f"**P** rises in {R['P']['tr']:.3f} s but overshoots by {R['P']['Mp']:.1f} % and stops "
                   f"{R['P']['ess_pct']:.1f} % short of the target. A proportional controller alone cannot remove "
                   "the steady-state error.")
    if stable["P"] and stable["PD"]:
        obs.append(f"**PD**: adding Kd changes the overshoot from {R['P']['Mp']:.1f} % to {R['PD']['Mp']:.1f} % and "
                   f"the settling time from {R['P']['ts']:.3f} s to {R['PD']['ts']:.3f} s. The steady-state error "
                   f"stays at {R['PD']['ess_pct']:.1f} %, because the derivative of a constant error is zero.")
    if stable["PI"]:
        obs.append(f"**PI**: the integral term removes the steady-state error ({zero(R['PI']['ess_pct']):.2f} %), but the "
                   f"response is slower (settles in {R['PI']['ts']:.3f} s) with {R['PI']['Mp']:.1f} % overshoot.")
    if stable["PID"]:
        obs.append(f"**PID** combines both effects: {zero(R['PID']['ess_pct']):.2f} % steady-state error, "
                   f"{R['PID']['Mp']:.1f} % overshoot and settling in {R['PID']['ts']:.3f} s.")
    if stable["MPC"]:
        vpk = {k: float(np.nanmax(np.abs(results[k]["sim"]["va"]))) for k in ALL}
        obs.append(f"**MPC** reaches the target with {zero(R['MPC']['ess_pct']):.2f} % steady-state error and "
                   f"{R['MPC']['Mp']:.1f} % overshoot, settling in {R['MPC']['ts']:.3f} s, while never exceeding "
                   f"±{cfg['mpc_Vmax']:g} V (peak {vpk['MPC']:.1f} V). The classical controllers ask for up to "
                   f"{max(vpk[k] for k in CONTROLLERS):,.0f} V, which a ±{cfg['mpc_Vmax']:g} V drive could not supply.")
    for o in obs:
        st.markdown("- " + o)

    st.subheader("Conclusion")
    if best:
        m = R[best]
        st.success(f"**{best} is the most suitable controller for this motor** with the current settings: "
                   f"rise time {m['tr']:.3f} s, settling time {m['ts']:.3f} s, overshoot {m['Mp']:.1f} %, "
                   f"steady-state error {zero(m['ess_pct']):.2f} %.")
        if best_limited and best_limited != best:
            ml = R[best_limited]
            st.info(f"**Within the ±{cfg['mpc_Vmax']:g} V supply limit, {best_limited} is the most suitable**: it is "
                    f"the best controller whose voltage never exceeds the limit (settling {ml['ts']:.3f} s, overshoot "
                    f"{ml['Mp']:.1f} %, steady-state error {zero(ml['ess_pct']):.2f} %).")
        st.caption("Chosen by: stable; then steady-state error within 2 %; then overshoot within 10 %; then the "
                   "shortest settling time.")
    else:
        st.error("None of the controllers is stable with the current gains.")
