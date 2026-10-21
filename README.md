# DC Motor Speed Control: P, PD, PI, PID and MPC

A simulation study of classical controllers for the speed of an armature-controlled DC motor. The motor is modelled by its transfer function, the four controllers are simulated for a step in reference speed, and their rise time, settling time, overshoot and steady-state error are compared. A Model Predictive Controller (MPC) with a voltage limit and speed limits is added and compared with them.

The web app lets anyone change the motor parameters (including the inductance L) and each controller's gains, and see the speed response, the three state variables (θ, ω, iₐ) and the comparison update instantly. All calculations are in `core/`; users only see the front end.

## Run it

**Online:** upload this folder to a public GitHub repository, then deploy it at [share.streamlit.io](https://share.streamlit.io) (Create app → choose the repository, branch `main`, file `app.py` → Deploy).

**On your computer:**

```bash
cd motorctrl_lab
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # opens http://localhost:8501
```

## The app

| Tab | Contents |
| --- | --- |
| 📘 Overview | What the lab does, the block diagram, the four controllers in one line each, how to use the app |
| ⚙️ Motor model | Motor equations, transfer function G(s), state-space model with the three state variables, motor poles and time constants |
| 🎛 Controllers | Controller equations, selected gains, closed-loop transfer functions, poles and stability |
| 🔮 MPC | What MPC does, the prediction model, the optimisation problem, its matrix (QP) form, current settings, and a plot of one plan versus what happens |
| 📈 Step responses | Speed response of all four controllers, the three state variables, the armature voltage |
| 📊 Comparison | Table and charts of rise time, settling time, overshoot and steady-state error; observations; the most suitable controller |

Sidebar: motor parameters (R, L, J, B, Kₐ, K_b), gains for each controller, MPC settings (horizon N, sample time Ts, weights q and r, voltage limit, speed limits), reference speed and simulation time, and a button to return to the selected design values.

## Model

    Vₐ = R·iₐ + L·diₐ/dt + K_b·ω
    J·dω/dt + B·ω = Kₐ·iₐ

    G(s) = Ω(s)/Vₐ(s) = Kₐ / [ JL·s² + (BL + RJ)·s + (RB + KₐK_b) ]

Default motor: R = 1 Ω, L = 0.5 H, J = 0.01 kg·m², B = 0.1 N·m·s/rad, Kₐ = K_b = 0.01.

## Selected gains and results (defaults)

| Controller | Gains | Rise time (s) | Settling time (s) | Overshoot (%) | Steady-state error (%) |
| --- | --- | --- | --- | --- | --- |
| P | Kp = 100 | 0.099 | 0.567 | 25.1 | 9.1 |
| PD | Kp = 100, Kd = 5 | 0.095 | 0.326 | 5.3 | 9.1 |
| PI | Kp = 50, Ki = 100 | 0.164 | 0.809 | 16.4 | 0 |
| PID | Kp = 100, Ki = 200, Kd = 10 | 0.105 | 0.268 | 0.9 | 0 |

PID is the most suitable controller for this motor.

## Files

```
app.py              web front end
requirements.txt    Python packages
assets/             block-diagram images
core/config.py      parameters, defaults and selected gains
core/model.py       transfer function, state space, closed loop, stability
core/simulate.py    closed-loop step-response simulation
core/metrics.py     rise time, settling time, overshoot, steady-state error, best controller
core/mpc.py         MPC: prediction matrices, quadratic program (OSQP), closed-loop simulation
core/plots.py       charts
```
