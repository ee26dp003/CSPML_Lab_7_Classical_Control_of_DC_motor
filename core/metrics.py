"""Step-response performance metrics and the comparison verdict."""

import numpy as np

from .config import CONTROLLERS


def _cross(t, y, level):
    i = np.flatnonzero(y >= level)
    if not len(i):
        return np.nan
    i = i[0]
    if i == 0:
        return t[0]
    return t[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (t[i] - t[i - 1])


def step_metrics(sim):
    """Rise time 10–90 %, settling time ±2 %, overshoot, steady-state error."""
    t, w, r = sim["t"], sim["omega"], sim["ref"]
    if sim["blew_up"]:
        return dict(tr=np.nan, ts=np.nan, Mp=np.nan, y_final=np.nan, ess=np.nan, ess_pct=np.nan, settled=False)
    k = max(5, len(w) // 50)
    yf = float(np.mean(w[-k:]))
    tr = _cross(t, w, 0.9 * yf) - _cross(t, w, 0.1 * yf)
    Mp = max(0.0, (np.max(w) - yf) / abs(yf) * 100) if yf else np.nan
    out = np.flatnonzero(np.abs(w - yf) > 0.02 * abs(yf))
    ts = 0.0 if not len(out) else (t[out[-1] + 1] if out[-1] + 1 < len(t) else np.nan)
    return dict(tr=float(tr), ts=float(ts), Mp=float(Mp), y_final=yf, ess=r - yf, ess_pct=100 * (r - yf) / r,
                settled=bool(np.isfinite(ts) and ts < 0.85 * t[-1]))


def best_controller(results):
    """Stable first; then steady error within 2 %; then overshoot within 10 %; then fastest settling."""
    ranked = []
    for k in CONTROLLERS:
        R = results[k]
        m = R["metrics"]
        if not R["cl"]["stable"] or R["sim"]["blew_up"]:
            continue
        ranked.append((abs(m["ess_pct"]) > 2, m["Mp"] > 10, m["ts"] if np.isfinite(m["ts"]) else 1e9, k))
    return sorted(ranked)[0][3] if ranked else None
