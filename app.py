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
from core.config import (CONTROLLER_NAMES, CONTROLLERS, DESIGN_GAINS, GAIN_INFO, MOTOR, N_FILTER,  # noqa: E402
                         TEST, default_config)
from core.metrics import best_controller, step_metrics  # noqa: E402
from core.model import closed_loop, motor, motor_summary, state_space  # noqa: E402
from core.simulate import simulate  # noqa: E402

st.set_page_config(page_title="DC Motor Speed Control", page_icon="⚙️", layout="wide")

# ------------------------------------------------------------------ settings state
if "cfg_R" not in st.session_state or st.session_state.pop("reset", False):
    for k, v in default_config().items():
        st.session_state["cfg_" + k] = v


def num_input(key, label, lo, hi, step, unit, help_):
    st.number_input(f"{label} [{unit}]", min_value=float(lo), max_value=float(hi), step=float(step),
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


results = run_all(tuple(sorted(cfg.items())))
p = motor(cfg)
M = motor_summary(p)
best = best_controller(results)


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
    return ", ".join([f"Kp = {g['Kp']:g}"] + ([f"Ki = {g['Ki']:g}"] if "I" in k else [])
                     + ([f"Kd = {g['Kd']:g}"] if "D" in k else []))


def zero(v, tol=0.05):
    """Treat tiny leftovers (still settling, numerical) as zero."""
    return 0.0 if np.isfinite(v) and abs(v) < tol else v


def fmt(v, d=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


# ------------------------------------------------------------------ header
st.title("Classical speed control of a DC motor")
st.caption("A simulation study of P, PD, PI and PID controllers on an armature-controlled DC motor.")

tabs = st.tabs(["📘 Overview", "⚙️ Motor model", "🎛 Controllers", "📈 Step responses", "📊 Comparison"])

# ================================================================== overview
with tabs[0]:
    st.subheader("What this lab does")
    st.markdown("""
We want a DC motor to spin at exactly the speed we ask for, reach it quickly, and not overshoot. The only thing
we can change is the **armature voltage Vₐ**. A **controller** measures the speed, compares it with the target
speed ω\\*, and turns the difference (the **error** e = ω\\* − ω) into a voltage. This is called closed-loop or
feedback control.

We model the motor as a transfer function, try the four classical controllers, simulate how the speed responds
when the target suddenly jumps (a **step input**), and compare them.
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

**How to use this app**
1. Set the motor values and controller gains in the sidebar on the left (the defaults are the selected design).
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
    st.caption(f"In the simulation the derivative is built as Kd·N·s/(s+N) with N = {N_FILTER:g} rad/s, a "
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

# ================================================================== responses
with tabs[3]:
    show = st.multiselect("Controllers to show", CONTROLLERS, default=CONTROLLERS)
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
        st.plotly_chart(plots.voltage(results, show), width="stretch")
        st.caption("The voltage each controller applies to the motor. A derivative term gives a large, very short "
                   "spike at the moment the target jumps.")

# ================================================================== comparison
with tabs[4]:
    st.subheader("Comparison of performance metrics")
    rows = []
    for k in CONTROLLERS:
        m, cl = results[k]["metrics"], results[k]["cl"]
        ok = cl["stable"] and not results[k]["sim"]["blew_up"]
        rows.append({"Controller": k, "Gains": gains_text(k, cl["gains"]),
                     "Rise time (s)": m["tr"] if ok else np.nan, "Settling time (s)": m["ts"] if ok else np.nan,
                     "Overshoot (%)": m["Mp"] if ok else np.nan, "Final speed (rad/s)": m["y_final"] if ok else np.nan,
                     "Steady-state error (%)": zero(m["ess_pct"]) if ok else np.nan,
                     "Stable": "yes" if ok else "no"})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in df.columns
                                if c not in ("Controller", "Gains", "Stable")})
    st.plotly_chart(plots.metric_bars(results, CONTROLLERS), width="stretch")
    st.download_button("⬇ Download table (CSV)", df.to_csv(index=False), file_name="controller_comparison.csv")

    st.subheader("Observations")
    obs = []
    R = {k: results[k]["metrics"] for k in CONTROLLERS}
    stable = {k: results[k]["cl"]["stable"] and not results[k]["sim"]["blew_up"] for k in CONTROLLERS}
    for k in CONTROLLERS:
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
    for o in obs:
        st.markdown("- " + o)

    st.subheader("Conclusion")
    if best:
        m = R[best]
        st.success(f"**{best} is the most suitable controller for this motor** with the current settings: "
                   f"rise time {m['tr']:.3f} s, settling time {m['ts']:.3f} s, overshoot {m['Mp']:.1f} %, "
                   f"steady-state error {zero(m['ess_pct']):.2f} %.")
        st.caption("Chosen by: stable; then steady-state error within 2 %; then overshoot within 10 %; then the "
                   "shortest settling time.")
    else:
        st.error("None of the controllers is stable with the current gains.")
