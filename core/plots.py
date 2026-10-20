"""Plotly charts. One fixed colour per controller everywhere."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COL = {"P": "#2a78d6", "PD": "#eb6834", "PI": "#1baf7a", "PID": "#eda100"}
GRID = "rgba(128,128,128,0.18)"
MUTED = "rgba(128,128,128,0.7)"


def _style(fig, height, title=None, hover="x unified"):
    fig.update_layout(height=height, title=dict(text=title, font=dict(size=15)) if title else None,
                      margin=dict(l=60, r=20, t=60 if title else 30, b=50), hovermode=hover,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _thin(a, n=2500):
    return a[::max(1, len(a) // n)]


def _label(k, g):
    parts = [f"Kp={g['Kp']:g}"] + ([f"Ki={g['Ki']:g}"] if "I" in k else []) + ([f"Kd={g['Kd']:g}"] if "D" in k else [])
    return f"{k} ({', '.join(parts)})"


def speed(results, show):
    fig = go.Figure()
    first = results[show[0]]["sim"] if show else next(iter(results.values()))["sim"]
    fig.add_hline(y=first["ref"], line=dict(color=MUTED, dash="dash"), annotation_text="reference ω*",
                  annotation_position="bottom right")
    for k in show:
        S, g = results[k]["sim"], results[k]["cl"]["gains"]
        fig.add_trace(go.Scatter(x=_thin(S["t"]), y=_thin(S["omega"]), name=_label(k, g),
                                 line=dict(color=COL[k], width=2.4), hovertemplate="%{y:.4f} rad/s"))
    fig.update_xaxes(title_text="time [s]")
    fig.update_yaxes(title_text="speed ω [rad/s]")
    return _style(fig, 440, "Closed-loop speed response to a step in reference speed")


def states(results, show):
    fig = make_subplots(3, 1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=["x₁ = θ  shaft position [rad]", "x₂ = ω  shaft speed [rad/s]",
                                        "x₃ = iₐ  armature current [A]"])
    for k in show:
        S = results[k]["sim"]
        for row, key, unit in ((1, "theta", "rad"), (2, "omega", "rad/s"), (3, "ia", "A")):
            fig.add_trace(go.Scatter(x=_thin(S["t"]), y=_thin(S[key]), name=k, legendgroup=k,
                                     showlegend=row == 1, line=dict(color=COL[k], width=2),
                                     hovertemplate=f"%{{y:.4g}} {unit}"), row, 1)
    fig.update_xaxes(title_text="time [s]", row=3, col=1)
    _style(fig, 720, None)
    fig.update_layout(margin=dict(t=70), legend=dict(y=1.06))
    return fig


def voltage(results, show):
    """Armature voltage. The very short derivative spike at t = 0 is cut off so the rest stays readable."""
    fig = go.Figure()
    lo, hi, peak = np.inf, -np.inf, 0.0
    for k in show:
        S = results[k]["sim"]
        fig.add_trace(go.Scatter(x=_thin(S["t"]), y=_thin(S["va"]), name=k, line=dict(color=COL[k], width=2),
                                 hovertemplate="%{y:.4g} V"))
        v = S["va"][np.isfinite(S["va"])]
        if len(v):
            peak = max(peak, float(np.max(np.abs(v))))
            vv = S["va"][(S["t"] > 0.08) & np.isfinite(S["va"])] if "D" in k else v
            if len(vv):
                lo, hi = min(lo, vv.min()), max(hi, vv.max())
    title = "Armature voltage Vₐ applied by each controller"
    if np.isfinite(lo) and peak > 1.3 * max(abs(lo), abs(hi)):
        pad = 0.12 * (hi - lo)
        fig.update_yaxes(range=[lo - pad, hi + pad])
        title += f" (start-up spike of up to {peak:,.0f} V cut off)"
    fig.update_xaxes(title_text="time [s]")
    fig.update_yaxes(title_text="voltage Vₐ [V]")
    return _style(fig, 380, title)


def metric_bars(results, show):
    keys = [("tr", "Rise time [s]"), ("ts", "Settling time [s]"), ("Mp", "Overshoot [%]"),
            ("ess_pct", "Steady-state error [%]")]
    fig = make_subplots(1, 4, subplot_titles=[t for _, t in keys], horizontal_spacing=0.07)
    for i, (m, _) in enumerate(keys):
        vals = [results[k]["metrics"][m] for k in show]
        vals = [0.0 if (m == "ess_pct" and np.isfinite(v) and abs(v) < 0.05) else v for v in vals]
        vals = [abs(v) if np.isfinite(v) else None for v in vals]
        fig.add_trace(go.Bar(x=show, y=vals, marker_color=[COL[k] for k in show], showlegend=False,
                             text=["—" if v is None else f"{v:.3g}" for v in vals], textposition="outside",
                             cliponaxis=False, hovertemplate="%{x}: %{y:.4g}<extra></extra>"), 1, i + 1)
        fig.update_yaxes(rangemode="tozero", row=1, col=i + 1)
    _style(fig, 320, None, hover="closest")
    fig.update_layout(margin=dict(t=40, b=30), bargap=0.35, showlegend=False)
    return fig
